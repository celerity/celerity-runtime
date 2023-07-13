const React = require("react");

const CompLibrary = require("../../core/CompLibrary.js");

const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

function formatDOI(doi) {
  if (doi == null) return "";
  return `  \n**DOI**:&nbsp;[${doi}](https://dx.doi.org/${doi})`;
}

function formatVid(vid) {
  if (vid == null) return "";
  return `  \n**[Video](vid)**`;
}

const publications = [
  {
    title: "Command Horizons: Coalescing Data Dependencies while Maintaining Asynchronicity",
    authors:
      "Peter Thoman, Philip Salzmann",
    publishedAt: "WAMTA 2023",
    doi: "10.1007/978-3-031-32316-4_2"
  },
  {
    title: "An Asynchronous Dataflow-Driven Execution Model For Distributed Accelerator Computing",
    authors:
      "Philip Salzmann, Fabian Knorr, Peter Thoman, Philipp Gschwandtner, Biagio Cosenza, and Thomas Fahringer",
    publishedAt: "CCGRID 2023",
    doi: "10.1109/CCGrid57682.2023.00018"
  },
  {
    title: "Declarative Data Flow in a Graph-Based Distributed Memory Runtime System",
    authors:
      "Fabian Knorr, Peter Thoman, Thomas Fahringer",
    publishedAt: "International Journal of Parallel Programming, 2022",
    doi: "10.1007/s10766-022-00743-4"
  },
  {
    title: "The Celerity High-level API: C++20 for Accelerator Clusters",
    authors:
      "Peter Thoman, Florian Tischler, Philip Salzmann, and Thomas Fahringer",
    publishedAt: "HLPP 2021",
  },
  {
    title:
      "Porting Real-World Applications to GPU Clusters: A Celerity and CRONOS Case Study",
    authors:
      "Philipp Gschwandtner, Ralf Kissmann, David Huber, Philip Salzmann, Fabian Knorr, Peter Thoman, and Thomas Fahringer",
    publishedAt: "eScience 2021",
  },
  {
    title: "Celerity: High-level C++ for Accelerator Clusters",
    authors: "Peter Thoman, Philip Salzmann, Biagio Cosenza, Thomas Fahringer",
    publishedAt: "Euro-Par 2019",
    doi: "10.1007/978-3-030-29400-7_21",
  },
];

const events = [
  {
    title: "SYCL and Celerity: High(-ish) level, vendor-independent C++ for GPU parallelism",
    authors: "Peter Thoman",
    info: "Guest Lecture, Trento 2023",
    video: "https://www.youtube.com/watch?v=xK_tCN9nm4Q"
  },
  {
    title: "Automatic Discovery of Collective Communication Patterns in Parallelized Task Graphs",
    authors: "Fabian Knorr",
    info: "HLPP 2023",
  },
  {
    title: "SYCL Panel Discussion",
    authors: "Peter Thoman",
    info: "IWOCL/SYCLcon 2021",
  },
  {
    title: "Celerity — High-Level Distributed Accelerator C++ Programming",
    authors: "Philipp Gschwandtner",
    info: "Talk at AHPC 2020",
  },
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

  const Research = () => (
    <div>
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
      </div>
      Celerity research is being carried out at the University of Innsbruck and the University of Salerno.<br/>
      This page provides an overview of selected research publications and talks.
    </div>
  );

  const Highlights = () => (
    <Block layout="twoColumn">
      {[
        {
          title: "Selected Publications",
          content: publications
            .map(
              ({ title, authors, publishedAt, doi }) =>
                `**${title}**  \n${authors}  \n**${publishedAt}**${formatDOI(doi)} \n\n`
            )
            .join("\n"),
        },
        {
          title: "Selected Talks & Demos",
          content: events
            .map(
              ({ title, authors, info, video }) =>
                `**${title}**  \n${authors}  \n**${info}**${formatVid(video)} \n\n`
            )
            .join("\n"),
        },
      ]}
    </Block>
  );

  const Acknowledgements = () => (
    <div>
      <h2>Acknowledgements</h2>
      <p>
        This project has received funding from the{" "}
        <strong>
          European High-Performance Computing Joint Undertaking (JU)
        </strong>{" "}
        under grant agreement <strong>No 956137</strong>.
      </p>

      <p>
        This research has been partially funded by the{" "}
        <strong>FWF (I 3388)</strong> and{" "}
        <strong>DFG (CO 1544/1-1, project number 360291326)</strong> as part of
        the <strong>CELERITY</strong> project.
      </p>
    </div>
  );

  return (
    <Container padding={["top", "bottom"]}>
      <div className="homeContainer">
        <div className="wrapper">
          <Research />
        </div>
      </div>
      <Highlights />
      <hr className="separator" />
      <div className="homeContainer">
        <div className="wrapper">
          <Acknowledgements />
        </div>
      </div>
    </Container>
  );
};

module.exports = ResearchPage;
