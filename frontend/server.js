import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "dist");
const types = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
};

createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
    const requestedPath = path.resolve(root, `.${pathname}`);
    const safePath = requestedPath.startsWith(`${root}${path.sep}`) || requestedPath === root;
    const candidate = safePath ? requestedPath : path.join(root, "index.html");
    const filePath = (await stat(candidate)).isDirectory() ? path.join(candidate, "index.html") : candidate;
    let content = await readFile(filePath);
    if (path.basename(filePath) === "index.html") {
      const runtimeConfig = `<script>window.RUNTIME_CONFIG=${JSON.stringify({
        VITE_API_URL: process.env.VITE_API_URL || "",
      }).replace(/</g, "\\u003c")};</script>`;
      content = Buffer.from(content.toString("utf8").replace("</head>", `${runtimeConfig}</head>`));
    }
    response.writeHead(200, { "Content-Type": types[path.extname(filePath)] || "application/octet-stream" });
    response.end(content);
  } catch {
    try {
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      response.end(await readFile(path.join(root, "index.html")));
    } catch {
      response.writeHead(404);
      response.end("Not found");
    }
  }
}).listen(Number(process.env.PORT || 4173), "0.0.0.0");
