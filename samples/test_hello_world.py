"""
Sample: Basic hello world test for ragsearch package.

This example demonstrates a simple test for the hello_world function.
"""
from ragsearch.hello_world import hello_world

def main():
    result = hello_world()
    print("hello_world() returned:", result)
    assert result == "Hello, world"

if __name__ == "__main__":
    main()
