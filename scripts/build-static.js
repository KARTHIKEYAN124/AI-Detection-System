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
const securityHeaders = {
  "content-security-policy": "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self' http://localhost:5000 http://127.0.0.1:5000; font-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'; form-action 'self'; upgrade-insecure-requests",
  "cross-origin-opener-policy": "same-origin",
  "permissions-policy": "camera=(), microphone=(), geolocation=(), payment=()",
  "referrer-policy": "strict-origin-when-cross-origin",
  "strict-transport-security": "max-age=31536000; includeSubDomains; preload",
  "x-content-type-options": "nosniff",
  "x-frame-options": "DENY"
};

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
        ...securityHeaders,
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store"
      }
    });
  }
};
`;

fs.writeFileSync(path.join(server, "index.js"), worker);
