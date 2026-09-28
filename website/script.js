// Exact excerpts and links from reviewed clip 348f0f69-cd49-4c00-af91-f1765155e858_1127-21.
const paths = {
  continue: {
    facts: [
      ["F3", "The mower has a rear collection bag attached."],
      ["F7", "I am already mid-pass with the lawnmower rather than setting it up or putting it away."],
    ],
    reasonId: "R3",
    reason: "The fact that the mower has a collection bag and mowing is ongoing gives me reason to continue mowing in an orderly pass, because that supports completing the yard work efficiently.",
    actionId: "A1",
    action: "Continue pushing the mower forward while keeping a straight, controlled line across the grass.",
  },
  steer: {
    facts: [
      ["F1", "I am pushing a gas-powered lawnmower forward across a residential lawn."],
      ["F6", "My hands stay on the mower handle while I walk at a steady pace."],
    ],
    reasonId: "R1",
    reason: "The fact that I am operating a powered mower while walking forward gives me reason to keep the mower under control and watch my path, because contact with uneven ground or obstacles could cause injury or loss of control.",
    actionId: "A2",
    action: "Slow slightly and steer the mower to stay clear of the tree and leaf-covered border.",
  },
  inspect: {
    facts: [
      ["F1", "I am pushing a gas-powered lawnmower forward across a residential lawn."],
      ["F6", "My hands stay on the mower handle while I walk at a steady pace."],
    ],
    reasonId: "R1",
    reason: "The fact that I am operating a powered mower while walking forward gives me reason to keep the mower under control and watch my path, because contact with uneven ground or obstacles could cause injury or loss of control.",
    actionId: "A3",
    action: "Pause briefly to inspect the ground ahead before making the next pass near the border.",
  },
};

const picker = document.getElementById("action-picker");
picker.hidden = false;
picker.addEventListener("change", (event) => {
  const path = paths[event.target.value];
  if (!path) return;
  const facts = path.facts.map(([id, text]) => {
    const paragraph = document.createElement("p");
    const label = document.createElement("b");
    label.textContent = id;
    paragraph.append(label, ` ${text}`);
    return paragraph;
  });
  document.getElementById("example-facts").replaceChildren(...facts);
  document.getElementById("example-reason").textContent = path.reason;
  document.getElementById("example-action").textContent = path.action;
  document.getElementById("reason-ref").textContent = `${path.reasonId} · supported by ${path.facts.map(([id]) => id).join(", ")}`;
  document.getElementById("action-ref").textContent = `${path.actionId} · supported by ${path.reasonId}`;
});

for (const button of document.querySelectorAll("[data-copy]")) {
  button.hidden = false;
  button.addEventListener("click", async () => {
    const source = document.getElementById(button.dataset.copy);
    const status = document.getElementById("copy-status");
    const label = button.textContent;
    try {
      await navigator.clipboard.writeText(source.textContent);
      button.textContent = "Copied";
      status.textContent = `${label === "Copy BibTeX" ? "Citation" : "Commands"} copied to clipboard.`;
    } catch {
      const range = document.createRange();
      range.selectNodeContents(source);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Text selected";
      status.textContent = "Clipboard unavailable. Text is selected; use your browser's copy command.";
    }
    window.setTimeout(() => { button.textContent = label; }, 2200);
  });
}

// The template's mobile drawer, without its paper-specific demo or search code.
document.body.classList.add("js");
const sidebar = document.getElementById("sidebar");
const menuButton = document.getElementById("mobileMenuBtn");
const overlay = document.getElementById("mobileOverlay");
const mobile = window.matchMedia("(max-width: 900px)");
menuButton.hidden = false;

function setMenu(open) {
  sidebar.classList.toggle("open", open);
  overlay.classList.toggle("open", open);
  sidebar.inert = mobile.matches && !open;
  menuButton.setAttribute("aria-expanded", String(open));
  menuButton.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
}

menuButton.addEventListener("click", () => setMenu(!sidebar.classList.contains("open")));
overlay.addEventListener("click", () => setMenu(false));
sidebar.addEventListener("click", (event) => {
  if (event.target.closest("a") && mobile.matches) setMenu(false);
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && sidebar.classList.contains("open")) {
    setMenu(false);
    menuButton.focus();
  }
});
mobile.addEventListener("change", () => setMenu(false));
setMenu(false);
