// Test script to check ruv-fann API
use ruv_fann;

fn main() {
    println!("ruv-fann version: {}", env!("CARGO_PKG_VERSION"));
    
    // Test what's available in ruv_fann
    println!("ruv_fann module contents:");
    
    // This will show compilation errors for what's NOT available
    // let network = ruv_fann::Network::new(&[4, 8, 4]);
    // let activation = ruv_fann::ActivationFunction::Sigmoid;
    // let training = ruv_fann::TrainingAlgorithm::Rprop;
}