# VINE

**VINE (Variational Inference with Node Embeddings)** is a program and set
of supporting libraries for variational inference of phylogenetic trees.

## Requirements

- CMake 3.x or later
- C/C++ compiler (GCC, Clang, etc.)
- [PHAST](https://github.com/CshlSiepelLab/phast) (Phylogenetic Analysis with Space/Time models)

## Building from Source

If PHAST is installed in a standard location, CMake will usually find it
automatically. Otherwise, you can specify its install prefix explicitly:

```bash
cmake -S . -B build \
   -DCMAKE_BUILD_TYPE=Release \
   -DPHAST_ROOT=/path/to/phast
cmake --build build
cmake --install build
```

Here, `PHAST_ROOT` should point to the installation prefix of PHAST (e.g.,
`/opt/homebrew/opt/phast`).

## Usage

For usage of the main vine executable and supporting programs, run them
with `--help`:

```bash
vine --help
```

## Support

For questions or bug reports, please use the [GitHub issue tracker](https://github.com/CshlSiepelLab/vine/issues).

## License

Both PHAST and VINE are distributed under the BSD 3-Clause License, a
permissive academic license that allows redistribution and modification
provided that attribution is retained. See [LICENSE](LICENSE) for details.
