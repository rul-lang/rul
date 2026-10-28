#!/usr/bin/env sh
# Official One-Line Installer for Rul (.rl) Toolchain (`rul`, `rul-mcp`, `rul-bench`)
# Usage: curl -fsSL https://raw.githubusercontent.com/rul-lang/rul/main/install.sh | sh
set -eu

REPO="rul-lang/rul"
REPO_URL="https://github.com/${REPO}.git"
INSTALL_DIR="${RUL_INSTALL_DIR:-$HOME/.local/bin}"

echo "==> Installing Rul (.rl) 128-Bit Binary Language Toolchain..."

OS="$(uname -s)"
ARCH="$(uname -m)"

if [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ] && command -v curl >/dev/null 2>&1 && command -v tar >/dev/null 2>&1; then
    TARBALL_URL="https://github.com/${REPO}/releases/latest/download/rul-linux-x86_64.tar.gz"
    TMP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMP_DIR"' EXIT
    if curl -fsSL "$TARBALL_URL" -o "$TMP_DIR/rul-linux-x86_64.tar.gz" 2>/dev/null; then
        mkdir -p "$INSTALL_DIR"
        tar -xzf "$TMP_DIR/rul-linux-x86_64.tar.gz" -C "$INSTALL_DIR"
        chmod +x "$INSTALL_DIR/rul" "$INSTALL_DIR/rul-mcp" "$INSTALL_DIR/rul-bench"
        echo "==> Installed prebuilt Linux x86_64 binaries to $INSTALL_DIR (Zero Rust/Cargo required):"
        echo "      $INSTALL_DIR/rul"
        echo "      $INSTALL_DIR/rul-mcp"
        echo "      $INSTALL_DIR/rul-bench"
        echo "==> Try creating a pure Rul project anywhere:"
        echo "      rul new hello_rul && cd hello_rul && rul run"
        exit 0
    fi
fi

if command -v cargo >/dev/null 2>&1; then
    cargo install --git "$REPO_URL" --bins --force
    echo "==> Installed from source via cargo: rul, rul-mcp, rul-bench"
    echo "==> Try creating a pure Rul project anywhere:"
    echo "      rul new hello_rul && cd hello_rul && rul run"
else
    echo "Error: Could not download prebuilt binary for ${OS}/${ARCH} and cargo was not found." >&2
    exit 1
fi
