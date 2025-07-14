#!/usr/bin/env python3
"""
Build script for Neural Document Flow Python bindings.

This script handles building the Python wheel using maturin and
provides additional build options and validation.
"""

import argparse
import subprocess
import sys
import os
import shutil
from pathlib import Path


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def check_requirements():
    """Check if all build requirements are available."""
    print("Checking build requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"Error: Python 3.8+ required, got {python_version.major}.{python_version.minor}")
        return False
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check Rust
    try:
        result = run_command(["rustc", "--version"], check=False)
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
        else:
            print("Error: Rust not found. Install from https://rustup.rs/")
            return False
    except FileNotFoundError:
        print("Error: Rust not found. Install from https://rustup.rs/")
        return False
    
    # Check Cargo
    try:
        result = run_command(["cargo", "--version"], check=False)
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
        else:
            print("Error: Cargo not found")
            return False
    except FileNotFoundError:
        print("Error: Cargo not found")
        return False
    
    # Check maturin
    try:
        result = run_command(["maturin", "--version"], check=False)
        if result.returncode == 0:
            print(f"✓ {result.stdout.strip()}")
        else:
            print("Installing maturin...")
            run_command([sys.executable, "-m", "pip", "install", "maturin>=1.0,<2.0"])
    except FileNotFoundError:
        print("Installing maturin...")
        run_command([sys.executable, "-m", "pip", "install", "maturin>=1.0,<2.0"])
    
    print("✓ All requirements satisfied")
    return True


def clean_build():
    """Clean build artifacts."""
    print("Cleaning build artifacts...")
    
    # Remove target directory
    target_dir = Path("target")
    if target_dir.exists():
        shutil.rmtree(target_dir)
        print("✓ Removed target/")
    
    # Remove Python build artifacts
    for pattern in ["*.egg-info", "__pycache__", ".pytest_cache", "dist", "build"]:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                print(f"✓ Removed {path}")
            else:
                path.unlink()
                print(f"✓ Removed {path}")
    
    # Remove Python compiled files
    for py_file in Path(".").rglob("*.pyc"):
        py_file.unlink()
    
    print("✓ Build artifacts cleaned")


def build_wheel(release=True, target=None):
    """Build the Python wheel."""
    print(f"Building wheel (release={release})...")
    
    cmd = ["maturin", "build"]
    
    if release:
        cmd.append("--release")
    
    if target:
        cmd.extend(["--target", target])
    
    # Add feature flags
    cmd.extend(["--features", "default"])
    
    result = run_command(cmd)
    
    if result.returncode == 0:
        print("✓ Wheel built successfully")
        
        # Find the built wheel
        wheels_dir = Path("target/wheels")
        if wheels_dir.exists():
            wheels = list(wheels_dir.glob("*.whl"))
            if wheels:
                print(f"✓ Wheel location: {wheels[0]}")
                return wheels[0]
    
    return None


def develop_install():
    """Install in development mode."""
    print("Installing in development mode...")
    
    cmd = ["maturin", "develop"]
    result = run_command(cmd)
    
    if result.returncode == 0:
        print("✓ Development installation complete")
        return True
    
    return False


def run_tests():
    """Run the test suite."""
    print("Running tests...")
    
    # Python tests
    try:
        result = run_command([sys.executable, "-m", "pytest", "-v"], check=False)
        if result.returncode == 0:
            print("✓ Python tests passed")
        else:
            print("✗ Python tests failed")
            return False
    except FileNotFoundError:
        print("pytest not found, skipping Python tests")
    
    # Rust tests
    result = run_command(["cargo", "test"], check=False)
    if result.returncode == 0:
        print("✓ Rust tests passed")
    else:
        print("✗ Rust tests failed")
        return False
    
    return True


def validate_wheel(wheel_path):
    """Validate the built wheel."""
    print(f"Validating wheel: {wheel_path}")
    
    # Check wheel contents
    result = run_command([sys.executable, "-m", "zipfile", "-l", str(wheel_path)], check=False)
    if result.returncode != 0:
        print("✗ Failed to list wheel contents")
        return False
    
    # Try importing the module
    print("Testing wheel import...")
    test_env = os.environ.copy()
    test_env["PYTHONPATH"] = ""
    
    result = run_command([
        sys.executable, "-c",
        "import sys; sys.path.insert(0, '.'); import neuraldocflow; print('✓ Import successful')"
    ], check=False)
    
    if result.returncode == 0:
        print("✓ Wheel validation successful")
        return True
    else:
        print("✗ Wheel validation failed")
        return False


def create_distribution():
    """Create distribution files."""
    print("Creating distribution...")
    
    # Create dist directory
    dist_dir = Path("dist")
    dist_dir.mkdir(exist_ok=True)
    
    # Build wheel
    wheel_path = build_wheel(release=True)
    if not wheel_path:
        return False
    
    # Copy wheel to dist
    dist_wheel = dist_dir / wheel_path.name
    shutil.copy2(wheel_path, dist_wheel)
    print(f"✓ Wheel copied to {dist_wheel}")
    
    # Create source distribution (if possible)
    try:
        result = run_command(["maturin", "sdist"], check=False)
        if result.returncode == 0:
            print("✓ Source distribution created")
    except:
        print("Source distribution creation failed (optional)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Build Neural Document Flow Python bindings")
    parser.add_argument("--clean", action="store_true", help="Clean build artifacts")
    parser.add_argument("--develop", action="store_true", help="Install in development mode")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--wheel", action="store_true", help="Build wheel")
    parser.add_argument("--dist", action="store_true", help="Create distribution")
    parser.add_argument("--all", action="store_true", help="Clean, build, test, and create distribution")
    parser.add_argument("--release", action="store_true", default=True, help="Build in release mode")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    parser.add_argument("--target", help="Target triple for cross-compilation")
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Determine what to do
    if args.all:
        args.clean = True
        args.wheel = True
        args.test = True
        args.dist = True
    
    if not any([args.clean, args.develop, args.test, args.wheel, args.dist]):
        args.develop = True  # Default action
    
    success = True
    
    try:
        # Clean
        if args.clean:
            clean_build()
        
        # Build wheel
        if args.wheel:
            release = not args.debug
            wheel_path = build_wheel(release=release, target=args.target)
            if wheel_path:
                success &= validate_wheel(wheel_path)
            else:
                success = False
        
        # Development install
        if args.develop:
            success &= develop_install()
        
        # Run tests
        if args.test:
            success &= run_tests()
        
        # Create distribution
        if args.dist:
            success &= create_distribution()
        
        if success:
            print("\n✓ Build completed successfully!")
        else:
            print("\n✗ Build failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nBuild failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()