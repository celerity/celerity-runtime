const React = require("react");

const CompLibrary = require("../../core/CompLibrary.js");

const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const publications = [
  {
    title: "Celerity: High-level C++ for Accelerator Clusters",
    authors: "Peter Thoman, Philip Salzmann, Biagio Cosenza, Thomas Fahringer",
    publishedAt: "Euro-Par 2019",
  },
  {
    title:
      "CELERITY: Towards an Effective Programming Interface for GPU Clusters",
    authors:
      "Peter Thoman, Biagio Cosenza, Herbert Jordan, Philipp Gschwandtner, Thomas Fahringer, Ben Juurlink",
    publishedAt: "PDP 2018",
  },
];

const events = [
  {
    title: "Celerity: High-productivity Programming for Accelerator Clusters",
    authors: "Peter Thoman",
    info: "Talk at ScalPerf 2019",
  },
  {
    title: "Introducing Celerity: High-level C++ for Accelerator Clusters",
    authors: "Philip Salzmann",
    info: "Demo session at HPCS 2019",
  },
];

const ResearchPage = ({ config }) => {
  const { baseUrl } = config;

  const Block = (props) => (
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
            .join("\n"),
        },
        {
          title: "Selected Talks & Demos",
          content: events
            .map(
              ({ title, authors, info }) =>
                `### ${title}\n${authors}\n**${info}**`
            )
            .join("\n"),
        },
      ]}
    </Block>
  );

  const ProjectDescription = () => (
    <div>
      <p>
        Celerity is a research project that is being developed openly on{" "}
        <a href="https://github.com/celerity" target="_blank">
          GitHub
        </a>
        .
      </p>
      <p>
        If you want to contribute to Celerity's development, feel free to file a{" "}
        <a
          href="https://github.com/celerity/celerity-runtime/blob/master/CONTRIBUTING.md"
          target="_blank"
        >
          pull request
        </a>
        !<br /> Run into problems or limitations? Don't hesitate to{" "}
        <a
          href="https://github.com/celerity/celerity-runtime/issues/new"
          target="_blank"
        >
          open an issue
        </a>
        .<br /> For general questions and support, join us on our{" "}
        <a href="https://discord.gg/k8vWTPB">Discord server</a>!
      </p>
    </div>
  );

  const Research = () => (
    <div>
      <h2>Celerity Research</h2>
      For inquiries regarding research opportunities and collaboration, please
      contact either{" "}
      <a href="https://dps.uibk.ac.at/~petert/" target="_blank">
        Peter Thoman
      </a>{" "}
      at UIBK or{" "}
      <a href="https://www.cosenza.eu/" target="_blank">
        Biagio Cosenza
      </a>{" "}
      at UNISA.
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
    <Container padding={["bottom"]} className="contribute-page">
      <div className="homeContainer">
        <div className="wrapper">
          <img
            className="celerity-loves-github"
            src={`${baseUrl}img/celerity_loves_github.png`}
            srcSet={`${baseUrl}img/celerity_loves_github.png, ${baseUrl}img/celerity_loves_github@2x.png 2x`}
            alt=""
          />
          <ProjectDescription />
        </div>
      </div>
      <hr className="separator" />
      <div className="uni-logos">
        <a href="https://www.uibk.ac.at" target="_blank">
          <img
            src={`${baseUrl}img/uibk_logo.svg`}
            alt="University of Innsbruck"
          />
        </a>
        <a href="https://web.unisa.it" target="_blank">
          <img
            src={`${baseUrl}img/unisa_logo.svg`}
            alt="University of Salerno"
          />
        </a>
        <a href="https://www.tu-berlin.de" target="_blank">
          <img
            src={`${baseUrl}img/tub_logo.svg`}
            alt="Technical University of Berlin"
          />
        </a>
      </div>
      <div className="homeContainer">
        <div className="wrapper">
          <Research />
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
