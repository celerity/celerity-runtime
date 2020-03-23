const React = require("react");

const CompLibrary = require("../../core/CompLibrary.js");

const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const publications = [
  {
    title: "Celerity: High-level C++ for Accelerator Clusters",
    authors: "Peter Thoman, Philip Salzmann, Biagio Cosenza, Thomas Fahringer",
    publishedAt: "Euro-Par 2019"
  },
  {
    title:
      "CELERITY: Towards an Effective Programming Interface for GPU Clusters",
    authors:
      "Peter Thoman, Biagio Cosenza, Herbert Jordan, Philipp Gschwandtner, Thomas Fahringer, Ben Juurlink",
    publishedAt: "PDP 2018"
  }
];

const events = [
  {
    title: "Celerity: High-productivity Programming for Accelerator Clusters",
    authors: "Peter Thoman",
    info: "Talk at ScalPerf 2019"
  },
  {
    title: "Introducing Celerity: High-level C++ for Accelerator Clusters",
    authors: "Philip Salzmann",
    info: "Demo session at HPCS 2019"
  }
];

const ResearchPage = ({ config }) => {
  const { baseUrl } = config;

  const Block = props => (
    <GridBlock contents={props.children} layout={props.layout} />
  );

  const Highlights = () => (
    <Block layout="twoColumn">
      {[
        {
          title: "Publications",
          content: publications
            .map(
              ({ title, authors, publishedAt }) =>
                `### ${title}\n${authors}\n**${publishedAt}**`
            )
            .join("\n")
        },
        {
          title: "Selected Talks & Demos",
          content: events
            .map(
              ({ title, authors, info }) =>
                `### ${title}\n${authors}\n**${info}**`
            )
            .join("\n")
        }
      ]}
    </Block>
  );

  const ProjectDescription = () => (
    <div>
      Celerity is a joint research project by the{" "}
      <a href="https://www.uibk.ac.at" target="_blank">
        University of Innsbruck
      </a>{" "}
      and the{" "}
      <a href="https://www.tu-berlin.de" target="_blank">
        Technical University of Berlin
      </a>
      .
    </div>
  );

  const Acknowledgements = () => (
    <div>
      <h2>Acknowledgements</h2>
      This research has been partially funded by the{" "}
      <strong>FWF (I 3388)</strong> and{" "}
      <strong>DFG (CO 1544/1-1, project number 360291326)</strong> as part of
      the <strong>CELERITY</strong> project.
    </div>
  );

  return (
    <Container padding={["bottom", "top"]}>
      <div className="uni-logos">
        <img
          src={`${baseUrl}img/uibk_logo.svg`}
          alt="University of Innsbruck"
        />
        <img
          src={`${baseUrl}img/tub_logo.svg`}
          alt="Technical University of Berlin"
        />
      </div>
      <div className="homeContainer">
        <div className="wrapper homeWrapper">
          <ProjectDescription />
        </div>
      </div>
      <Highlights />
      <div className="homeContainer">
        <div className="wrapper">
          <Acknowledgements />
        </div>
      </div>
    </Container>
  );
};

module.exports = ResearchPage;
