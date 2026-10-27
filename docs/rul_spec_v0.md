# Rul Language & `.rl` Binary Execution Specification (v0.1)

- **Status**: APPROVED (v0.1 Production Baseline — Columnar Data Pipeline Domain)
- **File Extension**: `.rl`
- **Core Equation**:
  $$\text{Artifact (}\mathtt{.rl}\text{)} = \text{Algorithm} + \text{Schedule} + \text{Proof}$$
  $$\text{Engine} = \text{Linear Verifier } O(N) + \text{Tiered Stencil/Fuser Engine } O(N)$$

---

## 1. Phạm vi & Bất biến Kiến trúc (Scope & Architectural Invariants)

1. **Binary-First Single Source of Truth**:
   - Mọi chương trình Rul thực thi đều ở định dạng nhị phân phẳng `.rl`, căn chỉnh theo biên 16-byte/64-byte để nạp trực tiếp vào bộ nhớ (`Zero-Copy`).
   - Core Engine tuyệt đối không chứa Lexer hay Parser văn bản. Mọi chuyển đổi từ biểu diễn ký hiệu của LLM (Symbolic/JSON/DSL) sang `.rl` được thực hiện bởi tầng công cụ cầu nối (**Rul Transpiler/Packer**) trước khi đưa vào Engine.
2. **Tất định & Kết thúc Hữu hạn (Deterministic Termination)**:
   - Đồ thị lệnh trong `.rl` là một **Topological SSA-DAG** (Đồ thị có hướng không chu trình dạng gán đơn tĩnh).
   - Mọi node thứ $i$ (`0 <= i < NodeCount`) chỉ được phép tham chiếu các node đầu vào có chỉ số `< i`. Bất biến này đảm bảo chương trình không thể có vòng lặp vô hạn ẩn và cho phép kiểm chứng toàn bộ đồ thị trong **đúng 1 vòng quét tiến $O(N)$**.
3. **Tách biệt `Safety Verification` và `Intent Verification`**:
   - **Lớp 1 — Safety Verification ($O(N)$ Static Check)**: Chứng minh 100% không có lỗi Use-After-Consume, không có xung đột Đọc/Ghi (Data Race / Aliasing), không tràn mảng (Buffer Overflow) và không vượt quyền phần cứng (`CapabilityMask`).
   - **Lớp 2 — Intent Verification (`AssertContract` + Tier 0 Golden Oracle)**: Kiểm chứng tính đúng đắn của kết quả đầu ra trên tập dữ liệu mẫu hoặc bất biến giá trị ngay ở biên chương trình.

---

## 2. Định dạng Nhị phân `.rl` (Binary Layout)

Một file hoặc vùng nhớ `.rl` gồm 4 phần nối tiếp nhau:

```text
+-----------------------------------------------------------------------+
| 1. RulHeader (64 bytes, 64-byte aligned)                              |
+-----------------------------------------------------------------------+
| 2. ArenaDescriptor Table (arena_count * 16 bytes)                     |
+-----------------------------------------------------------------------+
| 3. ProofEntry Sidecar Table (proof_count * 16 bytes)                  |
+-----------------------------------------------------------------------+
| 4. NodeFrame Table (node_count * 16 bytes — 128-bit Instruction DAG)  |
+-----------------------------------------------------------------------+
```

### 2.1. `RulHeader` (64 bytes)

| Offset (Byte) | Trường | Kiểu | Mô tả |
| :--- | :--- | :--- | :--- |
| `0..8` | `magic` | `[u8; 8]` | Hằng số định danh: `b"RUL\x00\x01\x00\x00\x00"` (v0.1) |
| `8..16` | `graph_hash` | `u64` | Mã băm toàn vẹn 64-bit (FNV-1a / BLAKE3-64) của phần thân đồ thị |
| `16..24` | `capability_mask` | `u64` | Mặt nạ quyền hạn phần cứng (`0x00` = Pure Compute, `0x01` = NetRing, `0x02` = StorageDMA, `0x04` = GpuDispatch) |
| `24..28` | `arena_count` | `u32` | Số lượng vùng nhớ cột `Arena` được khai báo |
| `28..32` | `proof_count` | `u32` | Số lượng chứng chỉ biên trong `ProofSidecarTable` |
| `32..36` | `node_count` | `u32` | Số lượng khung lệnh 128-bit trong `NodeFrame Table` |
| `36..40` | `output_node` | `u32` | Chỉ số node kết xuất chính của đồ thị |
| `40..64` | `reserved` | `[u8; 24]` | Dự trữ (bắt buộc bằng `0`) |

### 2.2. `ArenaDescriptor` (16 bytes)

Mỗi vùng nhớ dữ liệu đầu vào hoặc vùng nhớ cấp phát mới đều là một cột đồng nhất kiểu (**SoA Column**):

| Offset | Trường | Kiểu | Mô tả |
| :--- | :--- | :--- | :--- |
| `0..4` | `arena_id` | `u32` | Chỉ số định danh `Arena` (`0 .. arena_count - 1`) |
| `4..8` | `element_count` | `u32` | Số phần tử trong cột (`Shape Length`) |
| `8` | `type_lane` | `u8` | Kiểu dữ liệu cơ sở & độ rộng SIMD (xem mục 2.4) |
| `9` | `mutability` | `u8` | `0 = Immutable Input`, `1 = Linear Mutable Region` |
| `10..12` | `alignment` | `u16` | Căn chỉnh bộ nhớ theo byte (mặc định `64` cho AVX-512 / Cache Line) |
| `12..16` | `initial_token` | `u32` | Phiên bản `AffineToken` khởi tạo cho `Arena` này |

### 2.3. `ProofEntry` (16 bytes — Sidecar Refinement Table)

Trong domain xử lý dữ liệu dạng cột, 90% node sử dụng `proof_idx = 0` (`TRIVIAL_BY_SHAPE` — tự động an toàn vì cùng độ dài cột). Bảng `ProofEntry` chỉ được tra cứu khi có phép cắt lát (`ArenaSlice`) hoặc truy cập gián tiếp (`Gather`):

| Offset | Trường | Kiểu | Mô tả |
| :--- | :--- | :--- | :--- |
| `0..2` | `rule_kind` | `u16` | `1 = SliceBounds (lo <= hi <= len)`, `2 = IndexInRange (max_idx < target_len)` |
| `2..4` | `target_arena` | `u16` | Chỉ số `Arena` đích được chứng minh |
| `4..8` | `bound_lo` | `u32` | Cận dưới không âm |
| `8..12` | `bound_hi` | `u32` | Cận trên (bắt buộc `<= ArenaDescriptor.element_count`) |
| `12..16` | `witness_node` | `u32` | Node cung cấp điều kiện kẹp biên (`Clamp` / `Mask`) nếu có |

### 2.4. Khung Lệnh 128-bit `NodeFrame` (16 bytes)

```text
Byte:  0          1            2            3            4..7 (32-bit)
     +----------+------------+------------+------------+------------------------+
     | opcode   | type_lane  | schedule   | proof_idx  | token_op (32 bits)     |
     | (u8)     | (u8)       | _hint (u8) | (u8)       | [mode:2b|id:14b|ver:16]|
     +----------+------------+------------+------------+------------------------+
     | in_a (u32)                         | in_b (32)                          |
     +------------------------------------+-------------------------------------+
Byte:  8..11 (32-bit)                       12..15 (32-bit)
```

1. **`opcode (u8)`**:
   - `0x01` `ArenaInput`: Nạp cột dữ liệu từ `ArenaDescriptor[in_a]`.
   - `0x02` `ConstF64`: Hằng số thực dấu phẩy động (mã hóa từ `in_a` hoặc bits `in_a | in_b << 32`).
   - `0x03` `ArenaSlice`: Cắt lát vùng nhớ `in_a` theo chứng minh tại `proof_idx`.
   - `0x04` `ArenaMutate`: Ghi đè vào vùng nhớ khả biến, tiêu thụ `AffineToken` cũ và phát hành phiên bản mới.
   - `0x10` `Add`: Cộng hai cột cùng `Shape` (`out[i] = a[i] + b[i]`).
   - `0x11` `Sub`: Trừ hai cột cùng `Shape` (`out[i] = a[i] - b[i]`).
   - `0x12` `Mul`: Nhân hai cột cùng `Shape` (`out[i] = a[i] * b[i]`).
   - `0x13` `Fma`: Nhân-cộng hợp nhất hoặc tổ hợp tuyến tính.
   - `0x14` `CmpGt`: So sánh lớn hơn không nhánh (`mask[i] = a[i] > b[i]`).
   - `0x15` `SelectZero`: Chọn không nhánh theo mặt nạ (`out[i] = mask[i] ? val[i] : 0.0`).
   - `0x20` `FilterCompact`: Lọc và nén các phần tử thỏa mặt nạ `in_a` từ cột `in_b`.
   - `0x21` `ReduceSum`: Cộng dồn toàn bộ cột `in_a` thành một giá trị vô hướng.
   - `0x22` `ReduceMax`: Tìm giá trị lớn nhất trên cột `in_a`.
   - `0x23` `Gather`: Truy cập gián tiếp `target[indices[i]]` (bắt buộc có `proof_idx > 0` chứng minh `indices < target.len`).
   - `0x30` `RingEnqueue`: Đẩy kết quả ra hàng đợi phần cứng (yêu cầu bit `CAP_NET_RING` trong `Header.capability_mask`).
   - `0x40` `AssertContract`: Kiểm tra hợp đồng giá trị đầu ra.
2. **`type_lane (u8)`**:
   - 4 bit cao (`ElemType`): `0x1 = F64`, `0x2 = I64`, `0x3 = BoolMask`.
   - 4 bit thấp (`SimdWidth`): `0x1 = Scalar`, `0x4 = Lane4`, `0x8 = Lane8 (AVX-512 F64)`.
3. **`schedule_hint (u8)`**:
   - 4 bit cao (`fusion_group`): `0` = Thực thi độc lập (Unfused); `1..15` = Mã nhóm gộp vòng lặp (`Kernel Fusion Group`).
   - 4 bit thấp (`exec_policy`): `0 = AutoL1Tile`, `1 = SimdStream`, `2 = MultiCore`, `3 = GpuDma`.
4. **`token_op (u32)`**:
   - 2 bit cao (`TokenMode`):
     - `0b00` (`PureValue`): Không thay đổi trạng thái quyền sở hữu `Arena`.
     - `0b01` (`ReadLease`): Mượn quyền đọc chia sẻ trên `Arena[id]` tại phiên bản `ver`. Nhiều node có thể cùng giữ `ReadLease` miễn là `ver` khớp với phiên bản hiện hành của `Arena`.
     - `0b10` (`AffineConsume`): Tiêu thụ độc quyền `Arena[id]` tại phiên bản `ver` và nâng phiên bản của `Arena[id]` lên `ver + 1`. Mọi `ReadLease` hoặc `AffineConsume` sau đó nếu dùng lại `ver` cũ sẽ bị `Verifier` bác bỏ ngay lập tức (`UseAfterConsume`!).

---

## 3. Luật Kiểm chứng Tuyến tính $O(N)$ (`Verifier Rules`)

Bộ `Verifier` duyệt qua mảng `nodes[0..node_count]` đúng 1 lần, duy trì bảng trạng thái kích thước cố định theo `arena_count` và `node_count`:

1. **Luật Tô-pô (Topological SSA Rule)**:
   - Với mọi node tại chỉ số $i$ có đầu vào là chỉ số node (`in_a`, `in_b`), bắt buộc `in_a < i` và `in_b < i`.
2. **Luật Bảo toàn Quyền sở hữu (`AffineToken` & `ReadLease` Rule)**:
   - Mỗi `Arena[a]` khởi tạo với `current_version = initial_token`.
   - Khi một node yêu cầu `ReadLease(a, ver)`: kiểm tra `current_version[a] == ver`.
   - Khi một node yêu cầu `AffineConsume(a, ver)`: kiểm tra `current_version[a] == ver` VÀ `Arena[a].mutability == Mutable`, sau đó cập nhật `current_version[a] = ver + 1`.
3. **Luật Đồng nhất Hình học & Không tràn mảng (`Shape & Refinement Proof Rule`)**:
   - Với các phép toán cột 2 ngôi (`Add`, `Sub`, `Mul`, `CmpGt`, `SelectZero`): kiểm tra `shape_len(in_a) == shape_len(in_b)` (hoặc một trong hai là hằng vô hướng `ConstF64`).
   - Với `ArenaSlice` và `Gather`: bắt buộc `proof_idx` trỏ tới một mục hợp lệ trong `ProofEntry Table` có `bound_lo <= bound_hi <= target_arena.element_count`.
4. **Luật Sandbox Phần cứng (`Capability Rule`)**:
   - Với `RingEnqueue`: bắt buộc `(header.capability_mask & CAP_NET_RING) != 0`.

---

## 4. Kiến trúc Động cơ Thực thi 3 Tầng (3-Tier Execution Engine)

1. **Tier 0 — Reference Columnar Interpreter**:
   - Thực thi tuần tự từng node theo ngữ nghĩa toán học chuẩn để làm **Golden Oracle** kiểm chứng kết quả.
2. **Tier 1 — Vector Stencil Engine (`Copy-and-Patch` Columnar Kernels)**:
   - Áp các hàm mẫu vector hóa theo lô (`CHUNK = 8` phần tử `f64` tương đương thanh ghi 512-bit) trên từng node mà không cần biên dịch qua LLVM.
3. **Tier 2 — Whole-Graph Kernel Fuser**:
   - Đọc trường `schedule_hint.fusion_group` trên các node liên tiếp.
   - Khi phát hiện một nhóm gộp hợp lệ (ví dụ: chuỗi `ArenaInput → ConstF64 → CmpGt → Mul/Add → SelectZero → ReduceSum`), Engine hợp nhất toàn bộ nhóm thành **một vòng lặp thanh ghi SIMD duy nhất (Single-Pass Register Pipeline)** — đọc dữ liệu từ `Arena` đúng 1 lần, thực hiện đồng thời phép lọc không nhánh + biến đổi số học + cộng dồn trên các làn vector `f64x8`, xóa bỏ hoàn toàn việc cấp phát mảng trung gian trong RAM.
