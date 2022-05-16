#![recursion_limit = "512"]
mod feature;
mod functions;
mod processing;
mod util;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
