const React = require("react");

const CompLibrary = require("../../core/CompLibrary.js");

const MarkdownBlock = CompLibrary.MarkdownBlock;
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const HomeSplash = ({ siteConfig, language }) => {
  const { baseUrl, docsUrl } = siteConfig;
  const docsPart = `${docsUrl ? `${docsUrl}/` : ""}`;
  const langPart = `${language ? `${language}/` : ""}`;
  const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

  const SplashContainer = props => (
    <div className="homeContainer">
      <div className="homeSplashFade">
        <div className="wrapper homeWrapper">{props.children}</div>
      </div>
    </div>
  );

  const ProjectDescription = () => (
    <MarkdownBlock>
      Celerity aims to bring the power and ease of use of
      [SYCL](https://www.khronos.org/sycl) to distributed memory accelerator
      clusters.
    </MarkdownBlock>
  );

  const PromoSection = props => (
    <div className="section promoSection">
      <div className="promoRow">
        <div className="pluginRowBlock">{props.children}</div>
      </div>
    </div>
  );

  const Button = props => (
    <div className="pluginWrapper buttonWrapper">
      <a className="button" href={props.href} target={props.target}>
        {props.children}
      </a>
    </div>
  );

  return (
    <SplashContainer>
      <div className="celerity-logo">
        <img
          srcSet={`${baseUrl}img/celerity_logo.png 1x, ${baseUrl}img/celerity_logo@2x.png 2x`}
          src={`${baseUrl}img/celerity_logo.png`}
          alt="Celerity Logo"
        />
      </div>
      <div className="inner">
        <ProjectDescription />
        <PromoSection>
          <Button href={docUrl("getting-started")}>Get Started</Button>
        </PromoSection>
      </div>
    </SplashContainer>
  );
};

class Index extends React.Component {
  render() {
    const { config: siteConfig, language = "" } = this.props;
    const { baseUrl } = siteConfig;

    const Block = props => (
      <Container
        padding={["bottom", "top"]}
        id={props.id}
        background={props.background}
      >
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    const Features = () => (
      <Block layout="fourColumn">
        {[
          {
            title: "Built on SYCL",
            content:
              "Celerity is built on top of Khronos' emerging industry standard for accelerator programming.",
            image: `${baseUrl}img/sycl_logo.svg`,
            imageAlign: "top"
          },
          {
            title: "Designed for HPC",
            content:
              "Transparently run your workloads on large-scale accelerator clusters.",
            image: `${baseUrl}img/hpc_icon.svg`,
            imageAlign: "top"
          }
        ]}
      </Block>
    );

    const Disclaimer = () => (
      <div className="section celerity-disclaimer">
        <strong>Disclaimer:</strong> Celerity is a research project first and
        foremost, and is still in early development. While it does work for
        certain applications, it probably does not fully support your use case
        just yet. We'd however love for you to give it a try and tell us about
        how you could imagine using Celerity for your projects in the future.
      </div>
    );

    return (
      <div className="home-page">
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="mainContainer">
          <Features />
          <Disclaimer />
        </div>
      </div>
    );
  }
}

module.exports = Index;
