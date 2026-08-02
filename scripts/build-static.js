const fs = require("fs");
const path = require("path");

const root = process.cwd();
const dist = path.join(root, "dist");
const server = path.join(dist, "server");
const hostingDir = path.join(dist, ".openai");

fs.rmSync(dist, { recursive: true, force: true });
fs.mkdirSync(dist, { recursive: true });
fs.mkdirSync(server, { recursive: true });
fs.mkdirSync(hostingDir, { recursive: true });
fs.copyFileSync(path.join(root, "index.html"), path.join(dist, "index.html"));
const hosting = JSON.parse(fs.readFileSync(path.join(root, ".openai", "hosting.json"), "utf8"));
fs.writeFileSync(
  path.join(hostingDir, "hosting.json"),
  JSON.stringify({ project_id: hosting.project_id }, null, 2)
);
fs.copyFileSync(path.join(root, "favicon.svg"), path.join(dist, "favicon.svg"));

const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const favicon = fs.readFileSync(path.join(root, "favicon.svg"), "utf8");
const worker = `const html = ${JSON.stringify(html)};
const favicon = ${JSON.stringify(favicon)};

export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === "/health") {
      return Response.json({ status: "static", model_loaded: false });
    }
    if (url.pathname === "/favicon.svg") {
      return new Response(favicon, {
        headers: { "content-type": "image/svg+xml; charset=utf-8" }
      });
    }
    return new Response(html, {
      headers: {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store"
      }
    });
  }
};
`;

fs.writeFileSync(path.join(server, "index.js"), worker);
