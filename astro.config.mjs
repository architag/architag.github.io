// @ts-check
import { defineConfig } from "astro/config";

import react from "@astrojs/react";
import tailwind from "@astrojs/tailwind";

import { template } from "./src/settings";

import sitemap from "@astrojs/sitemap";

export default defineConfig({
    integrations: [react(), tailwind(), sitemap()],
    site: 'https://architag.github.io',
    output: 'static'
});