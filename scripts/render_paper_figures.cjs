"use strict";

const fs = require("fs");
const path = require("path");
const sharp = require("sharp");

const repoRoot = path.resolve(__dirname, "..");
const figureDir = path.join(repoRoot, "paper", "generated", "figures");

async function main() {
  const svgFiles = fs
    .readdirSync(figureDir)
    .filter((name) => name.toLowerCase().endsWith(".svg"))
    .sort();

  if (!svgFiles.length) {
    throw new Error(`No SVG figures found in ${figureDir}`);
  }

  for (const name of svgFiles) {
    const input = path.join(figureDir, name);
    const output = path.join(figureDir, name.replace(/\.svg$/i, ".png"));
    await sharp(input, { density: 192 }).png().toFile(output);
    console.log(`Rendered ${path.relative(repoRoot, output)}`);
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
