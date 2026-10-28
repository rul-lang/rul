//! In-Memory Model Context Protocol (MCP) Server Binary for Rul (`.rl`).
//!
//! Reads JSON-RPC 2.0 lines from `stdin` and writes responses to `stdout`
//! with sub-millisecond in-memory turnaround.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::indexing_slicing)]

use std::io::{self, BufRead, Write};

use rul::mcp::RulMcpServer;

fn main() {
    let server = RulMcpServer::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line_res in stdin.lock().lines() {
        let Ok(line) = line_res else {
            break;
        };
        let response = server.handle_line(&line);
        if !response.is_empty() {
            if writeln!(stdout, "{response}").is_err() {
                break;
            }
            if stdout.flush().is_err() {
                break;
            }
        }
    }
}
