# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic
Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Celerity should now be able to automatically assign a unique compute device
  to each node on a host, given that a sufficient number of devices is
  available. (9d3da06e)

### Fixed

- Don't print an error message regarding buffer lifetime for trivial programs
  (i.e., programs containing no tasks). (ae133458)

## [0.1.0] - 2019-09-05

__Hello, World!__ This is our first release!

