import { element } from "./common.js";

export function table(headers, rows) {
  const node = element("table"), head = element("thead"), tr = element("tr");
  for (const header of headers) tr.append(element("th", { scope: "col" }, header));
  head.append(tr); node.append(head); const body = element("tbody");
  for (const cells of rows) { const row = element("tr"); for (const cell of cells) { const td = element("td"); td.append(cell instanceof Node ? cell : document.createTextNode(String(cell))); row.append(td); } body.append(row); }
  node.append(body); return node;
}
export function labeled(label, control) {
  const wrapper = element("label", { class: "field" }); wrapper.append(element("span", { class: "field-label" }, label), control); return wrapper;
}
