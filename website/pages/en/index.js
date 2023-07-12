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

    const Example = () => (
      <div class="code-example">
      <MarkdownBlock class="code-example">
      {`\`\`\`cpp
#include <celerity.h>
using namespace celerity;

// naive, but distributed matrix-vector multiplication!
int main() {
    // (1) declare virtualized input and output buffers
    constexpr size_t size = 256;
    buffer<float, 2> matrix{{size, size}};
    buffer<float, 1> vector{{size}};
    buffer<float, 1> result{{size}};

    distr_queue q;
    q.submit([&](handler &cgh) {
        // (2) specify data access patterns to enable distributed execution
        accessor m(matrix, cgh, [size](chunk<1> chnk) {
            return subrange<2>({chnk.offset[0], 0}, {chnk.range[0], size});
        }, read_only);
        accessor v(vector, cgh, access::one_to_one(), read_only);
        accessor r(result, cgh, access::one_to_one(), write_only, no_init);

        // (3) launch the parallel computation
        cgh.parallel_for(range<1>(size), [=](item<1> item) {
            r[item] = 0;
            for (size_t i = 0; i < size; ++i) {
                r[item] += m[item.get_id(0)][i] * v[i];
            }
        });
    });
}
\`\`\``}
      </MarkdownBlock>
      </div>
    );

    return (
      <div className="home-page">
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="mainContainer">
          <Example />
          <Features />
        </div>
      </div>
    );
  }
}

module.exports = Index;
