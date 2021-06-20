#[cfg(feature = "timer")]
extern crate colored;

pub use self::inner::*;

#[cfg(feature = "timer")]
#[macro_use]
pub mod inner {
    pub use colored::Colorize;

    // print-trace requires std, so these imports are well-defined
    pub use std::{
        format, println,
        string::{String, ToString},
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };
    pub static NUM_INDENT: AtomicUsize = AtomicUsize::new(0);
    pub const PAD_CHAR: &'static str = "·";

    #[macro_export]
    macro_rules! timer_start {
        ($msg:expr) => {{
            use std::{sync::atomic::Ordering, time::Instant};
            use $crate::{compute_indent, Colorize, NUM_INDENT};

            let result = $msg();
            let start_info = "Start:".yellow().bold();
            let indent_amount = 2 * NUM_INDENT.fetch_add(0, Ordering::Relaxed);
            let indent = compute_indent(indent_amount);

            if NUM_INDENT.fetch_add(0, Ordering::Relaxed) <= 10 {
                println!("{}{:8} {}", indent, start_info, result);
            }
            NUM_INDENT.fetch_add(1, Ordering::Relaxed);
            (result, Instant::now())
        }};
    }

    #[macro_export]
    macro_rules! timer_end {
        ($time:expr) => {{
            use std::{io::Write, sync::atomic::Ordering};
            use $crate::{compute_indent, Colorize, NUM_INDENT};

            let time = $time.1;
            let final_time = time.elapsed();
            if let Ok(file_name) = std::env::var("BENCH_OUTPUT_FILE") {
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(file_name)
                    .unwrap();
                writeln!(&mut file, "{}, {:?}", $time.0, final_time).unwrap();
            }
            let final_time = {
                let secs = final_time.as_secs();
                let millis = final_time.subsec_millis();
                let micros = final_time.subsec_micros() % 1000;
                let nanos = final_time.subsec_nanos() % 1000;
                if secs != 0 {
                    format!("{}.{}s", secs, millis).bold()
                } else if millis > 0 {
                    format!("{}.{}ms", millis, micros).bold()
                } else if micros > 0 {
                    format!("{}.{}µs", micros, nanos).bold()
                } else {
                    format!("{}ns", final_time.subsec_nanos()).bold()
                }
            };

            let end_info = "End:".green().bold();
            let message = format!("{}", $time.0);

            if NUM_INDENT.fetch_add(0, Ordering::Relaxed) <= 10 {
                NUM_INDENT.fetch_sub(1, Ordering::Relaxed);
                let indent_amount = 2 * NUM_INDENT.fetch_add(0, Ordering::Relaxed);
                let indent = compute_indent(indent_amount);

                // Todo: Recursively ensure that *entire* string is of appropriate
                // width (not just message).
                println!(
                    "{}{:8} {:.<pad$}{}",
                    indent,
                    end_info,
                    message,
                    final_time,
                    pad = 75 - indent_amount
                );
            }
        }};
    }

    #[macro_export]
    macro_rules! add_to_trace {
        ($title:expr, $msg:expr) => {{
            use std::io::Write;
            use $crate::{
                compute_indent, compute_indent_whitespace, format, Colorize, Ordering, ToString,
                NUM_INDENT,
            };

            let start_msg = "StartMsg".yellow().bold();
            let end_msg = "EndMsg".green().bold();
            let title = $title();
            let start_msg = format!("{}: {}", start_msg, title);
            let end_msg = format!("{}: {}", end_msg, title);

            let start_indent_amount = 2 * NUM_INDENT.fetch_add(0, Ordering::Relaxed);
            let start_indent = compute_indent(start_indent_amount);

            let msg_indent_amount = 2 * NUM_INDENT.fetch_add(0, Ordering::Relaxed) + 2;
            let msg_indent = compute_indent_whitespace(msg_indent_amount);
            let mut final_message = "\n".to_string();
            for line in $msg().lines() {
                final_message += &format!("{}{}\n", msg_indent, line,);
            }
            if let Ok(file_name) = std::env::var("BENCH_OUTPUT_FILE") {
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(file_name)
                    .unwrap();
                writeln!(&mut file, "{}, {:?}", title, final_message).unwrap();
            }

            // Todo: Recursively ensure that *entire* string is of appropriate
            // width (not just message).
            println!("{}{}", start_indent, start_msg);
            println!("{}{}", msg_indent, final_message,);
            println!("{}{}", start_indent, end_msg);
        }};
    }

    pub fn compute_indent(indent_amount: usize) -> String {
        use std::env::var;
        let mut indent = String::new();
        let pad_string = match var("CLICOLOR") {
            Ok(val) => {
                if val == "0" {
                    " "
                } else {
                    PAD_CHAR
                }
            }
            Err(_) => PAD_CHAR,
        };
        for _ in 0..indent_amount {
            indent.push_str(&pad_string.white());
        }
        indent
    }

    pub fn compute_indent_whitespace(indent_amount: usize) -> String {
        let mut indent = String::new();
        for _ in 0..indent_amount {
            indent.push_str(" ");
        }
        indent
    }
}

#[cfg(not(feature = "timer"))]
#[macro_use]
mod inner {

    #[macro_export]
    macro_rules! timer_start {
        ($msg:expr) => {
            ()
        };
    }

    #[macro_export]
    macro_rules! add_to_trace {
        ($title:expr, $msg:expr) => {
            let _ = $msg;
        };
    }

    #[macro_export]
    macro_rules! timer_end {
        ($time:expr) => {
            let _ = $time;
        };
    }
}

mod tests {
    #[test]
    fn print_start_end() {
        let start = timer_start!(|| "Hello");
        add_to_trace!(|| "HelloMsg", || "Hello, I\nAm\nA\nMessage");
        timer_end!(start);
    }
}
