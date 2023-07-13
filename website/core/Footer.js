const React = require('react');

class Footer extends React.Component {
  docUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    const docsUrl = this.props.config.docsUrl;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    return `${baseUrl}${docsPart}${langPart}${doc}`;
  }

  pageUrl(doc, language) {
    const baseUrl = this.props.config.baseUrl;
    return baseUrl + (language ? `${language}/` : '') + doc;
  }

  render() {
    return (
      <footer className="nav-footer" id="footer">
        <section className="sitemap">
          <a href={this.props.config.baseUrl} className="nav-home">
            {this.props.config.footerIcon && (
              <img
                src={this.props.config.baseUrl + this.props.config.footerIcon}
                alt={this.props.config.title}
              />
            )}
          </a>
          <div>
            <h5>Docs</h5>
            <a href={this.docUrl('getting-started')}>
              Getting Started
            </a>
            <a href={this.docUrl('installation')}>
              Installation
            </a>
            <a href={this.docUrl('issues-and-limitations')}>
              Issues &amp; Limitations
            </a>
          </div>
          <div>
            <h5>Community</h5>
            <a href="https://discord.gg/k8vWTPB">Celerity Discord</a>
            <a
              href="https://stackoverflow.com/questions/tagged/celerity-hpc"
              target="_blank"
              rel="noreferrer noopener"
            >
              Stack Overflow
            </a>
          </div>
          <div>
            <h5>More</h5>
            <a href={this.pageUrl("research")}>Research</a>
            <a href={this.pageUrl("contribute")}>Contribute</a>
            <a
              href={this.props.config.repoUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
            <a
              className="github-button"
              href={this.props.config.repoUrl}
              data-icon="octicon-star"
              data-count-href="/facebook/docusaurus/stargazers"
              data-show-count="true"
              data-count-aria-label="# stargazers on GitHub"
              aria-label="Star this project on GitHub">
              Star
            </a>
          </div>
        </section>

        <section className="copyright">{this.props.config.copyright}</section>
        <section className="copyright">
          SYCL and the SYCL logo are trademarks of the Khronos Group Inc.
        </section>
      </footer>
    );
  }
}

module.exports = Footer;
