# NoRA project page

Static HTML using the [MINT paper-site template](https://github.com/mint-philosophy/b-r-minisite).
No frontend installation, build step, API keys, or deployment setup is needed.

From the repository root, run `make website` and open
<http://127.0.0.1:8766>. Only this directory is served. Stop the server with Ctrl-C.

## Editing

- `index.html`: paper metadata, benchmark explanation, commands, and citation.
- `styles.css`: NoRA content layout and local fonts.
- `script.js`: the example selector, copy buttons, and mobile menu.
- `template/`: the upstream stylesheet and theme CSS/JS, copied unchanged from
  revision `01e8558796ee77a9a141b7a8da69ba64a23c2974`. Normally leave these alone.
- `assets/mowing-frames.jpg`: pre-action montage for clip
  `348f0f69-cd49-4c00-af91-f1765155e858_1127-21`, from EgoNormia.
- `assets/fonts/`: locally served JetBrains Mono and Newsreader, with their
  SIL Open Font Licenses.

The sidebar, masthead, status bar, and light/dark themes reuse MINT's template.
Navigation is a short static list of NoRA sections. The shared masthead loads
CSS, JavaScript, and images from `mintresearch.org`; the rest is served locally.
If the masthead is unavailable, a text link remains. The page and first example
also remain readable without JavaScript. Light mode is the initial default.

The example is excerpted from `src/nora/assets/nora_test.jsonl`; it is not a full
annotation or a model response. Preserve its text, IDs, and support links when
editing. The static HTML contains the first selection as a no-JavaScript fallback.
The image's original source is the
[EgoNormia clip montage](https://huggingface.co/datasets/open-social-world/EgoNormia/resolve/main/video/348f0f69-cd49-4c00-af91-f1765155e858_1127-21/frame_all_prev.jpg).
Source media have separate access and usage terms from NoRA annotations.

Title, author order, and citation follow `CITATION.cff` and the
[NoRA preprint](https://arxiv.org/abs/2606.04806). The findings section summarizes
the paper.

## Deployment

Host only the `website/` directory. Confirm source-media redistribution terms,
retain the template and font attribution, and verify the dataset, code, and paper
links before deployment. Keep citation metadata consistent with `CITATION.cff`.

Check syntax with `node --check website/script.js` from the repository root.
Also check desktop/mobile layouts, the mobile menu, both themes, all three example
actions, and the copy buttons after content changes.
