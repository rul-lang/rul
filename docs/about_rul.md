Để hiểu rõ tại sao Rul (.rl) không phải là một thư viện hay một biến thể của Rust, mà là một giống loài ngôn ngữ hoàn toàn mới dành riêng cho AI Agents, hãy đi từ câu hỏi gốc rễ
  nhất: Một ngôn ngữ lập trình thực chất là cái gì?
  ──────
  ## 1. Bản chất: Ngôn ngữ cho Con người vs. Ngôn ngữ cho AI Agents

  • Ngôn ngữ truyền thống (C, Python, Go, Rust): Là giao thức dịch thuật giữa Đôi mắt + 10 ngón tay con người (vốn chỉ đọc hiểu được chữ cái tiếng Anh như if, for, let, { }) và Phần
  cứng Silicon (CPU/GPU). Vì phục vụ mắt người, 80% cấu trúc của Rust/C là "vỏ bọc văn bản" (Syntax Sugar, tên biến dài dòng, dấu chấm phẩy, suy luận kiểu ngầm định).
  • Ngôn ngữ AI-Native (Rul .rl): AI Agent không có mắt thịt và không có ngón tay gõ bàn phím. Bên trong não của LLM là các vector số học và đồ thị phụ thuộc. Bắt một AI Agent
  phải chuyển suy nghĩ của nó thành hàng nghìn ký tự ASCII kiểu pub fn process(data: &[Record]) -> Result<f64, Error> rồi lại bắt trình biên dịch đọc từng chữ cái đó để dựng lại đồ
  thị là một sự lãng phí trung gian vô nghĩa.

  Rul cắt bỏ toàn bộ lớp văn bản ở giữa. Trong Rul, một "chương trình" (file .rl) là một Đồ thị Nhị phân Phẳng (Flat Binary Graph) mà mỗi nút lệnh có kích thước cố định đúng 16
  bytes (128-bit), mang theo sẵn 3 thứ:

    Khung lệnh .𝚛𝚕 (128-bit) =     Algorithm     +             Schedule              +           Proof
                                   ╰───┬───╯                   ╰───┬──╯                          ╰─┬─╯
                               Làm phép tính gì?   Gộp vòng lặp / chạy SIMD thế nào?   Bằng chứng an toàn bộ nhớ

  │ 💡 Tại sao lúc tạo ra Rul ta lại nhắc tới Rust/C/Assembly?
  │ Giống như để chế tạo ra Cỗ máy in đầu tiên, người thợ phải dùng búa và đục bằng tay; hay trình biên dịch đầu tiên của Rust từng được viết bằng ngôn ngữ OCaml, trình biên dịch
  đầu
  │ tiên của Python được viết bằng C.
  │ Ta chỉ dùng khoảng ~1,500 dòng mã hệ thống cấp thấp (Rust repr(C)) để đóng ra "Cỗ máy thực thi Rul" (rul binary engine) đầu tiên. Khi cỗ máy rul này ra đời, AI Agent chỉ
  │ giao tiếp thuần túy bằng ngôn ngữ .rl!
  ──────
  ## 2. Cách tạo ra ngôn ngữ Rul (Quy trình chế tạo 4 Khối cốt lõi)

  Để tạo ra một ngôn ngữ truyền thống, người ta mất 3–5 năm viết: Lexer (tách từ) → Parser (cây cú pháp) → Type Inferencer (đoán kiểu) → Borrow Checker → LLVM IR Lowering → Linker.

  Để tạo ra Rul (.rl), chúng ta xóa bỏ 70% các bước trên và chỉ chế tạo đúng 4 khối cơ khí chính xác:

  ### Khối 1: Định nghĩa "Bảng tuần hoàn" 32 Opcodes & Khung nhị phân 16-byte (.rl Spec)

  Thay vì định nghĩa từ khóa tiếng Anh (while, match, struct), ta định nghĩa cấu trúc bộ nhớ của file .rl:

  • Header (64 bytes): Chứa chữ ký "RUL\x00\x01\x00\x00\x00", mã băm toàn vẹn BLAKE3, mặt nạ quyền phần cứng CapabilityMask (ví dụ: chương trình này có được phép đụng vào mạng hay đĩa
  cứng không?), và số lượng vùng nhớ Arena.
  • Mảng Khung lệnh 128-bit (NodeFrame - mỗi lệnh đúng 16 bytes):
    [ Opcode (8b) | TypeLane (8b) | ScheduleHint (8b) | ProofIdx (8b) | Out_Token (32b) | In_A (32b) | In_B (32b) ]

  • Tập lệnh chỉ gồm 32 Opcodes cấp cao xoay quanh 5 nguyên tử: cấp phát vùng nhớ cột (ArenaAlloc), biến đổi vector thuần khiết (PureMap: Add, Mul, CmpGt, Select...), luồng dữ liệu
  quan hệ (FilterCompact, ReduceSum, Gather, HashJoin), và I/O vòng (RingEnqueue).

  ### Khối 2: Chế tạo "Cổng Gác Toán Học" — Bộ Kiểm chứng Tuyến tính O(N) (Verifier)

  Đây là trái tim giúp Rul an toàn ngang ngửa hoặc hơn Rust nhưng kiểm tra xong trong < 1 mili-giây:

  • Trình nạp của Rul ánh xạ thẳng file .rl vào RAM bằng lệnh hệ thống mmap (tốn 0 ms đọc/tách chữ).
  • Bộ Verifier chỉ chạy đúng 1 vòng lặp for từ Node 0 đến Node N - 1 để kiểm tra 4 điều kiện toán học:
      1. Không chu trình ma: Chỉ số đầu vào In_A < Current_Node_Index (đảm bảo đồ thị luôn chạy tiến, không bao giờ treo vô hạn).
      2. Không xung đột bộ nhớ (AffineToken + Read-Lease): Mỗi vùng nhớ Arena có một thẻ bài (Token). Nhiều node muốn đọc song song thì dùng cờ Read-Lease; nhưng hễ một node muốn
      ghi đè hoặc hủy vùng nhớ đó, nó phải xuất trình đúng Token hiện hành và hủy thẻ cũ đi. Nếu AI dùng lại thẻ cũ → Từ chối ngay ở nano-giây đầu tiên!
      3. Không tràn mảng (Bounds-Safe by Construction): Vì các phép toán trong Rul chạy trên toàn bộ cột Arena<Shape>, nếu Node A cộng với Node B, Verifier chỉ cần so sánh
      Length(A) == Length(B). Nếu bằng nhau → Chắc chắn 100% không bao giờ tràn mảng ở runtime mà không cần lệnh if (i < len) ở từng phần tử!
      4. Không vượt quyền Sandbox (CapabilityMask): Nếu trong đồ thị có lệnh gửi mạng RingEnqueue nhưng Header không bật bit CAP_NET → Từ chối biên dịch!


  ### Khối 3: Chế tạo Động cơ Thực thi 3 Tầng (Tiered Execution Engine)

  Sau khi đi qua cổng Verifier, đồ thị .rl được thực thi qua 3 tầng:

  • Tầng 0 (Oracle Interpreter): Chạy thử trên 100 phần tử mẫu trong vài micro-giây để kiểm tra Hợp đồng Đầu vào/Đầu ra (AssertContract) — đảm bảo AI không chỉ viết code "an toàn bộ
  nhớ" mà còn đúng logic bài toán.
  • Tầng 1 (Copy-and-Patch Vector Stencil): Với mỗi Opcode trong .rl, động cơ dán thẳng một đoạn mã máy SIMD (AVX-512 / NEON) đã được tối ưu sẵn vào bộ nhớ thực thi. Biên dịch xong
  trong vài mili-giây.
  • Tầng 2 (Whole-Graph Kernel Fuser): Đọc byte ScheduleHint do AI gắn trên các Node có cùng FusionGroupID để gộp nhiều bước xử lý liên tiếp thành đúng 1 vòng lặp thanh ghi CPU/GPU
  duy nhất (dữ liệu chỉ đọc từ RAM lên thanh ghi 1 lần duy nhất, xóa sổ 90% thời gian chờ RAM!).

  ### Khối 4: Chế tạo Giao thức Giao tiếp với AI Agent (Assembler + Hardware Reward Loop)

  • Cung cấp cho AI Agent một Bộ đóng gói nhị phân (Rul Transpiler/Emitter) kèm luật khóa token (Constrained Decoding Schema) để AI phát ra trực tiếp các khung .rl hợp lệ.
  • Khi chương trình .rl chạy xong, Engine đọc các thanh ghi hiệu năng phần cứng của CPU (CPU Cycles, L1 Cache Misses, SIMD Utilization) và trả ngược về cho AI một gói tin 32-byte
  (Reward Vector). AI nhìn vào đó để đổi byte ScheduleHint nếu muốn vắt kiệt thêm tốc độ phần cứng.
  ──────
  ## 3. Rul (.rl) khác gì với Rust? (Giải phẫu Trực diện)

  Nhiều người hỏi: "Rust đã an toàn và nhanh rồi, tại sao AI không cứ thế mà viết Rust?"
  Câu trả lời nằm ở chỗ: Rust được thiết kế cho giới hạn sinh học của con người, nên nó trở thành cái cùm đối với AI Agent.

  1. Về Thời gian Suy luận & Biên dịch (Compile-time Bottleneck):
      • Trong Rust: Con người không thể tự viết chứng minh toán học cho từng biến, nên Rust bắt trình biên dịch phải tự suy luận (Type Inference, Trait Coherence, Borrow Checker) và
      bắt LLVM chạy hơn 100 vòng tối ưu hóa để đoán xem vòng lặp for tuần tự của con người có chuyển thành lệnh SIMD được không. Một dự án Rust lớn mất 30 giây – 5 phút để compile.
      Với một AI Agent cần thử nghiệm 100 phương án/giây, chờ đợi rustc 30 giây là "chết lâm sàng".
      • Trong Rul: AI mang theo lời giải (Proof + Schedule), Compiler chỉ kiểm tra trong O(N). Không có suy luận kiểu, không có nhân bản hàm generic (Zero Monomorphization), không
      có LLVM đoán mò. Kiểm chứng và nạp chạy mất < 1 ms - 15 ms.
  2. Về Cách Tổ chức Bộ nhớ & Tốc độ Thực thi (Runtime Bottleneck):
      • Trong Rust / C: Lập trình viên thường tạo Vec<Order> (Mảng các Struct — AoS). Khi chỉ cần lọc trên trường price, CPU vẫn phải kéo cả cục Order (gồm tên, địa chỉ, ngày tháng.
      ..) từ RAM vào bộ nhớ đệm L1 Cache, làm lãng phí 80% băng thông bộ nhớ. Ngoài ra, Rust mặc định chèn lệnh kiểm tra tràn mảng (Bounds Check) ở mọi thao tác arr[i].
      • Trong Rul: Cấu trúc dữ liệu duy nhất là Arena theo Cột (SoA — Structure of Arrays) căn chỉnh đúng vạch 64-byte của CPU Cache Line, và mọi phép toán đều là Vector/Tensor
      cấp cao. Cộng với việc gộp nhân (Kernel Fusion) và 0% kiểm tra biên lúc chạy (do đã chứng minh ở bước Verifier), Rul khai thác sát giới hạn vật lý của băng thông RAM và tập
      lệnh SIMD.

  ──────
  ## 4. Cách sử dụng Rul như thế nào? (Ví dụ Thực tế từ đầu đến cuối)

  Hãy xem cùng một yêu cầu thực tế giao cho AI Agent:

  │ Bài toán: "Trên 100 triệu giao dịch tài chính, hãy lọc các giao dịch có amount > 1000, tính phí rủi ro bằng amount * amount, rồi tính tổng toàn bộ."

  ### Cách 1: Khi AI Agent dùng ngôn ngữ truyền thống (Rust)

  AI phải sinh ra một đống chữ ASCII, rồi gọi cargo build --release (đợi vài giây), và nếu viết ngây thơ sẽ tạo ra cấp phát bộ nhớ trung gian hoặc trượt cache:

    // AI tốn ~120 tokens văn bản ASCII cho hàm và khung chương trình
    pub fn compute_risk_total(amounts: &[f64], threshold: f64) -> f64 {
        amounts
            .iter()
            .copied()
            .filter(|&x| x > threshold) // Nhánh rẽ (Branch misprediction trên CPU!)
            .map(|x| x * x)
            .sum()
    }

  ### Cách 2: Khi AI Agent sử dụng Rul (.rl)

  Quy trình diễn ra trong 3 bước hoàn toàn tự động (tổng thời gian từ lúc AI phát lệnh đến lúc chạy xong trên 100M phần tử chỉ tính bằng mili-giây):

  #### Bước A — AI Agent phát ra Đồ thị 6 Khung lệnh 128-bit (.rl)

  Khi dùng qua cầu nối Rul Symbolic Emitter (hoặc phát thẳng khung nhị phân), toàn bộ chương trình chỉ gồm 6 Nodes (tương đương 96 bytes lệnh + 64 bytes Header = 160 bytes file
  .rl!):

    # Biểu diễn Disassembly (để con người kiểm toán) của file pipeline.rl (160 bytes):
    HEADER: Magic="RUL v0.1" | CapMask=0x00 (Pure Compute Sandbox) | Arenas=1 | Nodes=6

    NODE 0: [Op=ArenaInput  | Type=f64x8 | Sched=Group1_AVX | Proof=0 | Out=Tok(A0,Read)]  InA=Arena(0)  InB=0
    NODE 1: [Op=ConstScalar | Type=f64x8 | Sched=Group1_AVX | Proof=0 | Out=Val(N1)     ]  InA=1000.0    InB=0
    NODE 2: [Op=CmpGt       | Type=f64x8 | Sched=Group1_AVX | Proof=0 | Out=Mask(N2)    ]  InA=Node(0)   InB=Node(1)
    NODE 3: [Op=Mul         | Type=f64x8 | Sched=Group1_AVX | Proof=0 | Out=Val(N3)     ]  InA=Node(0)   InB=Node(0)
    NODE 4: [Op=SelectZero  | Type=f64x8 | Sched=Group1_AVX | Proof=0 | Out=Val(N4)     ]  InA=Node(2)   InB=Node(3)
    NODE 5: [Op=ReduceSum   | Type=f64x8 | Sched=Group1_AVX | Proof=0 | Out=OutCommit   ]  InA=Node(4)   InB=0

  Hãy nhìn vào sự khác biệt vật lý ở 6 Node này:

  1. Không có nhánh rẽ (if/else) làm nghẽn CPU: AI dùng CmpGt + SelectZero (Branchless SIMD Predication) và đánh dấu tất cả thuộc Sched = Group1_AVX.
  2. Gộp toàn bộ vào 1 vòng lặp duy nhất (Zero RAM Round-trip): Nhờ cả 6 Node đều mang nhãn Group1_AVX, Kernel Fuser của Rul thực thi phép so sánh, nhân bình phương, chọn theo mặt
  nạ và cộng dồn ngay bên trong thanh ghi SIMD của CPU trong đúng 1 lần quét qua RAM. Không có mảng tạm nào được tạo ra!

  #### Bước B — Nạp, Kiểm chứng & Thực thi qua CLI hoặc Shared Memory API

  Con người hoặc AI Orchestrator sử dụng công cụ rul với 3 lệnh cực kỳ đơn giản:

    # 1. Đóng gói từ biểu diễn cấu trúc của AI sang file nhị phân .rl (160 bytes)
    rul pack pipeline.rl.json -o pipeline.rl

    # 2. Kiểm toán an toàn (Disassemble + Verify O(N) trong < 0.01 ms)
    rul verify pipeline.rl
    # -> Output: [VERIFIED] 6 nodes, 0 memory aliasing, 0 bounds risk, Capabilities=NONE (Time: 420 ns)

    # 3. Thực thi trực tiếp file .rl với phản hồi phần cứng cho AI Agent
    rul run pipeline.rl --input transactions.arena --tier 2 --telemetry-json
    # -> Output: {"result": 4.998e17, "verify_ns": 420, "exec_ms": 18.4, "bandwidth_gbps": 43.4, "l1_miss_rate": 0.002}

  Hoặc trong môi trường Production, AI Agent không cần ghi ra file đĩa cứng mà truyền thẳng mảng byte .rl (160 bytes) qua bộ nhớ chia sẻ (rul::execute_bytes(&rl_bytes, &arenas))
  để chạy ngay lập tức trong cùng tiến trình!
  ──────
  ## 5. Bảng Đối chiếu Chi tiết Kiến trúc: C vs. Go vs. Rust vs. Rul (.rl)

   Chiều Kiến trúc                   │ C (gcc/clang -O3)                 │ Go (gc)                           │ Rust (rustc + LLVM)               │ Rul (.rl — AI-Native Language)
  ───────────────────────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────────────────────┼───────────────────────────────────
   1. Đối tượng thiết kế gốc         │ Con người gõ bàn phím (1972)      │ Nhóm kỹ sư con người (2009)       │ Kỹ sư hệ thống con người (2015)   │ 100% AI Agents & Autonomous Loops
                                     │                                   │                                   │                                   │ (2026+)
   2. Đơn vị mã nguồn cơ sở          │ Văn bản ASCII (.c, .h)            │ Văn bản UTF-8 (.go)               │ Văn bản UTF-8 (.rs)               │ File Nhị phân Phẳng .rl (Mảng các
                                     │                                   │                                   │                                   │ khung lệnh cố định 128-bit / 16-
                                     │                                   │                                   │                                   │ byte)
   3. Chi phí Đọc & Phân tích cú     │ Cao (Đọc hàng trăm file .h, tiền  │ Trung bình (Parser 1 pha trên văn │ Rất cao (Macro expansion, Syn/AST │ Bằng 0 (mmap thẳng file .rl vào
   pháp (Lex/Parse)                  │ xử lý #include, dựng AST)         │ bản)                              │ phức tạp)                         │ RAM, 0% Syntax Error về mặt toán
                                     │                                   │                                   │                                   │ học)
   4. Cơ chế Đặt tên & Định danh     │ Chuỗi ký tự tên biến/hàm + Linker │ Chuỗi ký tự + Package path        │ Chuỗi ký tự + Mangled Symbol cực  │ Chỉ số mảng phẳng u32 (nội cục) +
                                     │ symbol dài                        │                                   │ dài                               │ Mã băm nội dung BLAKE3 64-bit
                                     │                                   │                                   │                                   │ (liên module)
   5. Cơ chế An toàn Bộ nhớ          │ Không có (Con trỏ thô, dễ bị      │ Garbage Collector (GC) lúc chạy   │ Borrow Checker (Trình biên dịch   │ Explicit AffineToken + Read-Lease
                                     │ Segfault, Use-After-Free)         │ (Tốn CPU & gây khựng độ trễ)      │ tự giải hệ ràng buộc phức tạp lúc │ (AI gắn thẻ quyền sở hữu sẵn,
                                     │                                   │                                   │ compile)                          │ Verifier kiểm tra bằng phép so
                                     │                                   │                                   │                                   │ sánh O(1) mỗi node)
   6. Kiểm tra Tràn mảng (Bounds     │ Không kiểm tra (Gây lỗ hổng       │ Kiểm tra bằng lệnh if ở mọi lần   │ Kiểm tra bằng lệnh if lúc chạy    │ 0% kiểm tra lúc chạy (Bounds-Safe
   Checking)                         │ Buffer Overflow)                  │ truy cập lúc chạy                 │ (trừ khi LLVM chứng minh bỏ được) │ by Construction qua Arena<Shape>
                                     │                                   │                                   │                                   │ + bảng ProofIdx kiểm tra lúc nạp)
   7. Bố cục Bộ nhớ Mặc định (Memory │ AoS (Array of Structs) + malloc   │ Struct trên Heap có GC + con trỏ  │ AoS mặc định (Vec<Struct>), muốn  │ 100% Arena dạng Cột (SoA —
   Layout)                           │ rời rạc trên Heap → Trượt Cache   │ gián tiếp                         │ SoA phải tự viết tay rất cực      │ Structure of Arrays) căn chỉnh
                                     │ liên tục                          │                                   │                                   │ 64-byte Cache Line
   8. Chiến lược Tối ưu & Gộp vòng   │ Lập trình viên tự viết tay hoặc   │ Rất yếu (ít tự động vector hóa)   │ Phụ thuộc vào 100+ passes tìm     │ Schedule-Carrying Code: AI đính
   lặp (Scheduling)                  │ cầu nguyện LLVM đoán được         │                                   │ kiếm của LLVM (Tốn hàng chục giây │ kèm sẵn byte ScheduleHint (Gộp
                                     │                                   │                                   │ compile)                          │ nhóm nào, chạy SIMD hay GPU);
                                     │                                   │                                   │                                   │ Compiler chỉ áp khuôn trong O(N)
   9. Kỹ thuật Sinh mã máy (Codegen  │ LLVM / GIMPLE (Chậm, nặng nề)     │ Go SSA Backend (Nhanh vừa, mã máy │ LLVM Backend + Monomorphization   │ 3 Tầng: Tầng 0 (Oracle
   Backend)                          │                                   │ trung bình)                       │ (Nút thắt cổ chai làm compile cực │ Interpreter) → Tầng 1 (Copy-and-
                                     │                                   │                                   │ chậm)                             │ Patch SIMD Stencils <15ms) → Tầng
                                     │                                   │                                   │                                   │ 2 (Whole-Graph Kernel Fuser)
   10. Đảm bảo Chạy Đúng             │ Chạy thử mới biết (Dễ dính        │ Kiểm tra kiểu tĩnh cơ bản + Unit  │ Type-check chặt, nhưng vẫn có thể │ 2 Lớp khóa: (1) O(N) Verifier
   (Correctness)                     │ Undefined Behavior ngầm)          │ test viết tay                     │ sai logic nghiệp vụ               │ khóa 100% lỗi bộ nhớ; (2) Tier 0
                                     │                                   │                                   │                                   │ Oracle + AssertContract tự kiểm
                                     │                                   │                                   │                                   │ tra bất biến đầu ra trong <1 ms
   11. Quản lý Thư viện & Phụ thuộc  │ CMake / Makefile / .so hỗn loạn   │ go.mod (Văn bản)                  │ Cargo / crates.io (Hàng trăm      │ Content-Addressed Verified
                                     │                                   │                                   │ dependency văn bản phải compile   │ Intrinsics: Gọi trực tiếp các hàm
                                     │                                   │                                   │ lại từ đầu)                       │ chuẩn (Crypto, TLS, Sort) đã kiểm
                                     │                                   │                                   │                                   │ toán qua mã băm 64-bit bất biến
   12. Bảo mật & Quyền hạn Hệ thống  │ Toàn quyền tiến trình OS (Nếu bị  │ Toàn quyền tiến trình OS          │ Toàn quyền tiến trình OS (nếu có  │ Zero-Trust Capability ở mức Ngôn
   (Sandbox)                         │ lộ lỗ hổng là mất cả server)      │                                   │ dùng thư viện unsafe bên thứ 3)   │ ngữ: File .rl không thể gọi mạng
                                     │                                   │                                   │                                   │ hay đĩa cứng nếu CapabilityMask
                                     │                                   │                                   │                                   │ trên Header không cấp bit tương
                                     │                                   │                                   │                                   │ ứng
   13. Giao diện Báo lỗi             │ Văn bản dài cho người đọc trên    │ Văn bản cho người đọc             │ Văn bản dài giải thích chỗ sai    │ Binary Diagnostic Frame: Trả về
   (Diagnostics)                     │ Terminal                          │                                   │ kèm gạch chân ASCII               │ chính xác (NodeIdx, ErrorCode,
                                     │                                   │                                   │                                   │ ExpectedToken) để AI Agent mask
                                     │                                   │                                   │                                   │ logit và sửa ngay trong 1 bước
   14. Vòng lặp Tự tiến hóa Hiệu     │ Con người tự chạy perf / profiler │ Con người đọc pprof rồi sửa tay   │ Con người đọc flamegraph rồi sửa  │ Khép kín tự động: Engine trả về
   năng                              │ rồi ngồi sửa code mất cả tuần     │                                   │ tay                               │ Hardware Reward Vector (Cycles,
                                     │                                   │                                   │                                   │ L1 Miss, GB/s) sau mỗi lần chạy
                                     │                                   │                                   │                                   │ để AI tự đột biến ScheduleHint
                                     │                                   │                                   │                                   │ trong vài mili-giây
  ──────
  ### Bước tiếp theo

  Toàn bộ thiết kế trên đã được cụ thể hóa thành kế hoạch hành động với định dạng file .rl:

  1. Cập nhật lại bản tuyên ngôn new_language_for_AI.md và viết đặc tả kỹ thuật rul_spec_v0.md bao gồm đầy đủ bảng so sánh kiến trúc và cấu trúc khung 128-bit Algorithm + Schedule
  + Proof.
  2. Xây dựng ngay bộ Rul Core Engine (Bộ đóng gói/giải mã .rl, Cổng kiểm chứng O(N) Verifier, và Động cơ thực thi 3 tầng Tier 0 / Tier 1 / Tier 2) kèm bài Benchmark thực tế trên
  hàng chục triệu bản ghi để chứng minh bằng con số thật.
