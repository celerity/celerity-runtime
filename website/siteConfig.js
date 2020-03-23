const siteConfig = {
  title: "Celerity",
  tagline: "High-level C++ for Accelerator Clusters",
  url: "https://celerity.github.io",
  baseUrl: "/",

  projectName: "celerity.github.io",
  organizationName: "celerity",

  headerLinks: [
    { doc: "getting-started", label: "Docs" },
    { href: "https://github.com/celerity/celerity-runtime", label: "GitHub" },
    // { blog: true, label: "Blog" },
    { page: "research", label: "Research" }
  ],

  headerIcon: "img/celerity_icon.png",
  footerIcon: "img/celerity_icon.png",
  favicon: "img/favicon.ico",

  colors: {
    primaryColor: "rgb(245, 139, 105)",
    secondaryColor: "rgb(245, 139, 105)"
  },

  copyright: `Copyright © ${new Date().getFullYear()} Distributed and Parallel Systems Group, University of Innsbruck`,

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: "default"
  },

  algolia: {
    apiKey: "1945a24367750ada26456787eaa90102",
    indexName: "celerity"
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: ["https://buttons.github.io/buttons.js"],

  // On page navigation for the current documentation page.
  onPageNav: "separate",
  // No .html extensions for paths.
  cleanUrl: true,

  ogImage: "img/celerity_og_image.png",
  twitterImage: "img/celerity_twitter_image.png",

  // docsSideNavCollapsible: true,

  // enableUpdateBy: true,
  // enableUpdateTime: true,

  // Enable better syntax highlighting for C++
  usePrism: ["cpp"],

  // Custom config keys
  repoUrl: "https://github.com/celerity/celerity-runtime"
};

module.exports = siteConfig;
