# Contributing

Thank you for considering to contribute to Celerity!

In an effort to maintain a clean and consistent codebase over time, we have
collected a few guidelines that you should try and adhere to before
submitting a pull request in this repository. Not all of the things listed
below are hard-and-fast rules however, and you may follow them at your own
discretion. That being said, we do have a Continuous Integration (CI)
pipeline in place that will check pull request for the most important rules
outlined below.

## Code Style

Please use `standard_case` for all variables, class names etc. Template
arguments may use `PascalCase`.

In general, try to adhere to the coding style surrounding the section you are
modifying.

### Formatting Code Using Clang Format

All code should be formatted using `clang-format` before committing.
Unfortunately Clang Format frequently has breaking changes with major
releases (which are tied to the Clang release cycle), which means we have to
settle on a specific version.

We currently use `clang-format-8`. Please make sure to use the correct
version to avoid cluttering your diffs with unrelated format changes.
[Prebuilt binaries can be found on llvm.org](http://releases.llvm.org/8.0.0/).

You can automatically determine whether you've correctly formatted all of
your files using our [find-unformatted-files](ci/find-unformatted-files.sh)
script.

## Git Usage

While not always feasible, ideally all commits in our git history should
leave the project in a functioning state (i.e., it should compile and pass
all tests). To this end, try to squash work-in-progress commits into a single
commit before filing a PR. If possible, large change sets should however
still be split up into separate logical units, allowing for better
traceability later down the line.

Always use [`git rebase`](https://git-scm.com/docs/git-rebase) instead of
creating merge commits when incorporating upstream changes into your local
branches. This is to reduce noise in our git history.

Please use imperative form for your commit messages, i.e., prefer `"Add feature X"` over `"Added feature X"`, or just `"feature X"`. Furthermore, try
to come up with meaningful commit messages that succinctly describe your
change. If your commit introduces a large change, consider providing a more
detailed description in the commit message body.

## Running Tests

Please verify that all unit tests are passing after your change and consider
adding your own tests if possible. You can find all unit tests in the
[test](test) directory.

In addition to that, all code changes must pass our integration tests (which
currently are just the [examples](examples)). These will be executed
automatically by our CI pipeline using multiple GPU configurations. If you
want to try and run them yourself first, see the
[run-integration-tests](ci/run-integration-tests.sh) script.
