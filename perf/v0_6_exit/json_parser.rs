use std::io::{self, Read, Write};

const MAX_JSON_DEPTH: usize = 128;

const ERR_OK: i32 = 0;
const ERR_SYNTAX: i32 = 1;
const ERR_DEPTH: i32 = 2;
const ERR_UTF8: i32 = 3;

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
    obj_count: i64,
    arr_count: i64,
    str_count: i64,
    num_count: i64,
    bool_count: i64,
    null_count: i64,
    max_depth: i64,
}

impl<'a> Parser<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            obj_count: 0,
            arr_count: 0,
            str_count: 0,
            num_count: 0,
            bool_count: 0,
            null_count: 0,
            max_depth: 0,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                _ => break,
            }
        }
    }

    fn parse_value(&mut self, depth: usize) -> i32 {
        self.skip_ws();
        if self.pos >= self.input.len() {
            return ERR_SYNTAX;
        }
        match self.input[self.pos] {
            b'{' => self.parse_object(depth + 1),
            b'[' => self.parse_array(depth + 1),
            b'"' => self.parse_string(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            b't' => {
                if self.pos + 4 <= self.input.len() && &self.input[self.pos..self.pos + 4] == b"true"
                {
                    self.pos += 4;
                    self.bool_count += 1;
                    ERR_OK
                } else {
                    ERR_SYNTAX
                }
            }
            b'f' => {
                if self.pos + 5 <= self.input.len()
                    && &self.input[self.pos..self.pos + 5] == b"false"
                {
                    self.pos += 5;
                    self.bool_count += 1;
                    ERR_OK
                } else {
                    ERR_SYNTAX
                }
            }
            b'n' => {
                if self.pos + 4 <= self.input.len() && &self.input[self.pos..self.pos + 4] == b"null"
                {
                    self.pos += 4;
                    self.null_count += 1;
                    ERR_OK
                } else {
                    ERR_SYNTAX
                }
            }
            _ => ERR_SYNTAX,
        }
    }

    fn parse_object(&mut self, depth: usize) -> i32 {
        if depth > MAX_JSON_DEPTH {
            return ERR_DEPTH;
        }
        if (depth as i64) > self.max_depth {
            self.max_depth = depth as i64;
        }
        self.obj_count += 1;
        self.pos += 1; // '{'
        self.skip_ws();
        if self.pos < self.input.len() && self.input[self.pos] == b'}' {
            self.pos += 1;
            return ERR_OK;
        }
        loop {
            self.skip_ws();
            if self.pos >= self.input.len() || self.input[self.pos] != b'"' {
                return ERR_SYNTAX;
            }
            let err = self.parse_string();
            if err != ERR_OK {
                return err;
            }
            self.skip_ws();
            if self.pos >= self.input.len() || self.input[self.pos] != b':' {
                return ERR_SYNTAX;
            }
            self.pos += 1;
            let err = self.parse_value(depth);
            if err != ERR_OK {
                return err;
            }
            self.skip_ws();
            if self.pos >= self.input.len() {
                return ERR_SYNTAX;
            }
            match self.input[self.pos] {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    return ERR_OK;
                }
                _ => return ERR_SYNTAX,
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> i32 {
        if depth > MAX_JSON_DEPTH {
            return ERR_DEPTH;
        }
        if (depth as i64) > self.max_depth {
            self.max_depth = depth as i64;
        }
        self.arr_count += 1;
        self.pos += 1; // '['
        self.skip_ws();
        if self.pos < self.input.len() && self.input[self.pos] == b']' {
            self.pos += 1;
            return ERR_OK;
        }
        loop {
            let err = self.parse_value(depth);
            if err != ERR_OK {
                return err;
            }
            self.skip_ws();
            if self.pos >= self.input.len() {
                return ERR_SYNTAX;
            }
            match self.input[self.pos] {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    return ERR_OK;
                }
                _ => return ERR_SYNTAX,
            }
        }
    }

    fn parse_string(&mut self) -> i32 {
        self.pos += 1; // opening '"'
        let start = self.pos;
        while self.pos < self.input.len() {
            let b = self.input[self.pos];
            if b == b'"' {
                if std::str::from_utf8(&self.input[start..self.pos]).is_err() {
                    return ERR_UTF8;
                }
                self.pos += 1;
                self.str_count += 1;
                return ERR_OK;
            } else if b == b'\\' {
                self.pos += 1;
                if self.pos >= self.input.len() {
                    return ERR_SYNTAX;
                }
                match self.input[self.pos] {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => self.pos += 1,
                    b'u' => {
                        if self.pos + 4 >= self.input.len() {
                            return ERR_SYNTAX;
                        }
                        for k in 1..=4 {
                            let h = self.input[self.pos + k];
                            if !h.is_ascii_hexdigit() {
                                return ERR_SYNTAX;
                            }
                        }
                        self.pos += 5;
                    }
                    _ => return ERR_SYNTAX,
                }
            } else if b < 0x20 {
                return ERR_SYNTAX;
            } else {
                self.pos += 1;
            }
        }
        ERR_SYNTAX
    }

    fn parse_number(&mut self) -> i32 {
        if self.pos < self.input.len() && self.input[self.pos] == b'-' {
            self.pos += 1;
        }
        if self.pos >= self.input.len() {
            return ERR_SYNTAX;
        }
        match self.input[self.pos] {
            b'0' => self.pos += 1,
            b'1'..=b'9' => {
                self.pos += 1;
                while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                    self.pos += 1;
                }
            }
            _ => return ERR_SYNTAX,
        }
        if self.pos < self.input.len() && self.input[self.pos] == b'.' {
            self.pos += 1;
            if self.pos >= self.input.len() || !self.input[self.pos].is_ascii_digit() {
                return ERR_SYNTAX;
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < self.input.len()
            && (self.input[self.pos] == b'e' || self.input[self.pos] == b'E')
        {
            self.pos += 1;
            if self.pos < self.input.len()
                && (self.input[self.pos] == b'+' || self.input[self.pos] == b'-')
            {
                self.pos += 1;
            }
            if self.pos >= self.input.len() || !self.input[self.pos].is_ascii_digit() {
                return ERR_SYNTAX;
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        self.num_count += 1;
        ERR_OK
    }
}

fn main() {
    let mut input = Vec::new();
    io::stdin()
        .read_to_end(&mut input)
        .expect("failed to read stdin");

    let mut parser = Parser::new(&input);
    let mut err = parser.parse_value(0);
    if err == ERR_OK {
        parser.skip_ws();
        if parser.pos != input.len() {
            err = ERR_SYNTAX;
        }
    }

    let out = match err {
        ERR_OK => format!(
            "OK obj={} arr={} str={} num={} bool={} null={} max_depth={}\n",
            parser.obj_count,
            parser.arr_count,
            parser.str_count,
            parser.num_count,
            parser.bool_count,
            parser.null_count,
            parser.max_depth
        ),
        ERR_DEPTH => "ERR depth\n".to_owned(),
        ERR_UTF8 => "ERR utf8\n".to_owned(),
        _ => "ERR syntax\n".to_owned(),
    };
    io::stdout()
        .write_all(out.as_bytes())
        .expect("failed to write stdout");
}
