fn main() {
    let mime_type = "text/plain";
    let result = format!(
        "File type {} is blocked by security policy ", mime_type
    );
    println!("{}", result);
}