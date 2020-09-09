const React = require("react");

const CompLibrary = require("../../core/CompLibrary.js");

const Container = CompLibrary.Container;

// The former /research page is now called /contribute.
// FIXME: Once we are on Docusaurus v2, do a proper redirect.
const ResearchPage = () => {
  return (
    <Container padding={["bottom", "top"]}>
      <div>This page has moved to <a href="/contribute">Contribute</a>.</div>
    </Container>
  );
};

module.exports = ResearchPage;
