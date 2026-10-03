// Code-native engineering illustrations. All instances share these drawings;
// variants add internals that distinguish each registered equipment type.
const vessel = `<path class="pipe" d="M20 66H47 M111 43H139 M80 121V143"/><path class="support" d="M51 120L44 145H54L62 120 M101 120L108 145H118L110 120"/><path class="metal" d="M47 43Q47 22 80 22Q113 22 113 43V111Q113 133 80 133Q47 133 47 111Z"/><ellipse class="metal-dark" cx="80" cy="43" rx="33" ry="13"/><path class="highlight" d="M54 52V104"/><rect class="glass" x="72" y="57" width="17" height="51" rx="6"/><path class="liquid" d="M74 86H87V102Q80 108 74 102Z"/>`;
const column = `<path class="pipe" d="M16 81H59 M92 30H134V49 M92 123H134"/><path class="support" d="M58 127L53 146H65L69 127 M93 127L100 146H111L105 127"/><path class="metal" d="M59 29Q59 13 80 13Q101 13 101 29V125Q101 140 80 140Q59 140 59 125Z"/><ellipse class="metal-dark" cx="80" cy="30" rx="21" ry="8"/><path class="highlight" d="M65 42V120"/><path class="detail" d="M62 55H98 M62 74H98 M62 94H98 M62 113H98"/><rect class="glass" x="73" y="42" width="15" height="81" rx="4"/>`;
const motor = `<rect class="support" x="27" y="122" width="108" height="13" rx="3"/><rect class="metal-dark" x="91" y="64" width="41" height="49" rx="8"/><path class="detail" d="M99 70V106 M107 70V106 M115 70V106 M123 70V106"/>`;
const cooling = `<path class="cool" d="M131 23V49 M119 36H143 M122 27L140 45 M122 45L140 27"/>`;
const heating = `<path class="heat" d="M133 24Q118 37 128 49Q146 52 143 37Q137 41 133 24Z"/>`;
function illustration(type) {
  if (type === "Mixer")
    return (
      vessel +
      `<rect class="metal-dark" x="70" y="5" width="20" height="17" rx="3"/><path class="detail" d="M80 22V107 M62 79L98 87 M62 87L98 79 M21 92H47"/>`
    );
  if (type === "Splitter")
    return `<path class="pipe" d="M12 80H72V38H143 M72 80H143 M72 80V122H143"/><circle class="metal" cx="72" cy="80" r="22"/><circle class="metal-dark" cx="72" cy="80" r="12"/><path class="detail" d="M36 74V86 M124 32V44 M124 116V128"/>`;
  if (type === "Pump")
    return (
      motor +
      `<path class="pipe" d="M13 93H40 M64 57V31H99"/><circle class="metal" cx="62" cy="93" r="30"/><circle class="metal-dark" cx="62" cy="93" r="18"/><path class="detail" d="M62 78Q87 93 62 108Q40 93 62 78Z"/><rect class="metal" x="40" y="120" width="37" height="6"/>`
    );
  if (type === "Compressor" || type === "Expander")
    return (
      motor +
      `<path class="pipe" d="M12 81H39 M76 52V30H121"/><path class="metal" d="M39 61L81 48V112L39 100Z"/><path class="detail" d="M47 64V100 M59 60V104 M72 55V110"/>` +
      (type === "Expander"
        ? `<path class="heat" d="M15 34L30 43L15 52Z"/>`
        : `<path class="cool" d="M30 34L15 43L30 52Z"/>`)
    );
  if (type === "Valve")
    return `<path class="pipe" d="M12 93H148"/><path class="metal" d="M45 74L80 93L45 112Z M115 74L80 93L115 112Z"/><path class="detail" d="M80 90V44"/><ellipse class="metal-dark" cx="80" cy="38" rx="28" ry="9"/><path class="detail" d="M56 38H104 M80 30V46"/><rect class="metal" x="32" y="80" width="8" height="27"/><rect class="metal" x="121" y="80" width="8" height="27"/>`;
  if (type === "Pipe")
    return `<path class="pipe" d="M12 120H52V46H141"/><path class="metal" d="M43 97H61V106H43Z M99 37H108V55H99Z"/><path class="highlight" d="M15 117H49V43H138"/><path class="support" d="M48 121V146H68V139H56V121"/>`;
  if (type === "HeatExchanger")
    return `<path class="pipe" d="M10 80H30 M126 80H150 M45 58V30H69 M114 102V132H139"/><path class="support" d="M43 104L37 134H49L57 104 M108 104L115 134H127L121 104"/><rect class="metal" x="28" y="56" width="104" height="48" rx="22"/><ellipse class="metal-dark" cx="39" cy="80" rx="12" ry="24"/><path class="detail" d="M42 69H119 M42 80H125 M42 91H119"/><path class="highlight" d="M55 61H111"/>`;
  if (type === "Heater" || type === "Cooler")
    return (
      `<path class="pipe" d="M12 91H37 M122 91H149"/><rect class="metal-dark" x="36" y="43" width="88" height="84" rx="8"/><rect class="glass" x="47" y="55" width="66" height="60" rx="5"/><path class="detail" d="M54 69H101V81H59V94H103"/><path class="support" d="M44 127V143H53V127 M105 127V143H116V127"/>` +
      (type === "Heater" ? heating : cooling)
    );
  if (type === "Flash" || type === "Flash3" || type === "Decanter")
    return (
      vessel +
      (type === "Flash3"
        ? `<path class="liquid" d="M50 100H110V112Q80 132 50 112Z"/><path class="pipe" d="M113 108H141"/>`
        : type === "Decanter"
          ? `<rect class="heat" x="73" y="73" width="15" height="14"/><path class="pipe" d="M112 97H143"/>`
          : `<path class="detail" d="M56 66H104 M63 69L96 75"/>`)
    );
  if (/Distillation|Absorber|Stripper|Extractor/.test(type)) {
    let detail = "";
    if (/Distillation/.test(type))
      detail = `<path class="pipe" d="M128 31H145V73H101"/><rect class="cool" x="119" y="40" width="24" height="17" rx="3"/><ellipse class="heat" cx="126" cy="123" rx="15" ry="10"/>`;
    if (/Absorber/.test(type))
      detail = `<path class="cool" d="M32 30V46L26 41 M26 46H38"/><path class="detail" d="M76 49L85 61L76 70L85 81L76 93L85 104"/>`;
    if (/Stripper/.test(type))
      detail =
        heating +
        `<path class="detail" d="M76 109L85 98L76 86L85 74L76 63L85 50"/>`;
    if (/Extractor/.test(type))
      detail = `<path class="pipe" d="M20 115H59 M101 52H145"/><path class="liquid" d="M75 82H86V121H75Z"/>`;
    if (type.startsWith("Rigorous"))
      detail += `<path class="detail" d="M102 41H112V115H102 M109 50H117 M109 69H117 M109 88H117 M109 107H117"/>`;
    if (type === "McCabeThieleDistillation")
      detail += `<path class="detail" d="M22 52V29H42 M23 49L40 32"/>`;
    if (type === "CMODistillation")
      detail += `<path class="detail" d="M19 37H42 M19 46H42 M19 55H42"/>`;
    return column + detail;
  }
  if (type === "PFR" || type === "PackedBedReactor")
    return (
      `<path class="pipe" d="M13 79H31 M130 79H150"/><rect class="metal" x="30" y="53" width="101" height="52" rx="14"/><ellipse class="metal-dark" cx="37" cy="79" rx="10" ry="26"/><path class="support" d="M48 104V136H59V104 M105 104V136H117V104"/>` +
      (type === "PFR"
        ? `<path class="detail" d="M48 65H115V77H54V89H115"/>`
        : `<g class="heat">${[55, 73, 91, 109].map((x) => [65, 80, 95].map((y) => `<circle cx="${x}" cy="${y}" r="5"/>`).join("")).join("")}</g>`)
    );
  if (type === "MolecularSieveDryer")
    return (
      column +
      `<g class="heat">${[64, 82, 101, 119].map((y) => `<circle cx="78" cy="${y}" r="4"/>`).join("")}</g>` +
      cooling
    );
  if (type === "Filter")
    return `<path class="pipe" d="M12 68H32 M126 93H149"/><path class="support" d="M35 119V140H45V119 M112 119V140H122V119"/><rect class="metal-dark" x="29" y="51" width="104" height="69" rx="4"/>${[43, 57, 71, 85, 99, 113].map((x) => `<rect class="metal" x="${x}" y="46" width="7" height="81" rx="2"/>`).join("")}<path class="detail" d="M25 64H138 M25 111H138"/>`;
  if (type === "LayerCrystallizer")
    return (
      vessel +
      `<path class="cool" d="M53 61H61V109H53Z M99 61H107V109H99Z"/><path class="detail" d="M63 94L70 89L74 99L82 88L89 98L97 91"/>` +
      cooling
    );
  if (type === "Crystallizer")
    return (
      vessel +
      `<path class="detail" d="M60 97L69 83L78 101L89 86L99 102 M80 26V76"/>` +
      cooling
    );
  if (["Reactor", "EquilibriumReactor", "CSTR", "BatchReactor"].includes(type))
    return (
      vessel +
      `<rect class="metal-dark" x="69" y="6" width="22" height="17" rx="4"/><path class="detail" d="M80 23V104 M65 81L95 89 M65 89L95 81"/>` +
      (type === "EquilibriumReactor"
        ? `<path class="detail" d="M119 80H144L137 74 M144 90H119L126 96"/>`
        : type === "BatchReactor"
          ? `<circle class="metal-dark" cx="130" cy="94" r="15"/><path class="detail" d="M130 82V95H139"/>`
          : type === "CSTR"
            ? `<path class="pipe" d="M21 94H47 M114 83H145"/>`
            : heating)
    );
  throw new Error(`Missing equipment illustration: ${type}`);
}
let instance = 0;
export function equipmentSVG(type, attributes = "") {
  const id = `equipment-${++instance}`;
  const metal = `<linearGradient id="${id}-metal"><stop offset="0" stop-color="#355969"/><stop offset=".25" stop-color="#8eb4c0"/><stop offset=".48" stop-color="#c4dce1"/><stop offset=".72" stop-color="#789dab"/><stop offset="1" stop-color="#3e6476"/></linearGradient>`;
  const glass = `<linearGradient id="${id}-glass"><stop stop-color="#163b47"/><stop offset=".35" stop-color="#367782"/><stop offset=".65" stop-color="#285862"/><stop offset="1" stop-color="#153742"/></linearGradient>`;
  const art = illustration(type)
    .replaceAll('class="metal"', `class="metal" style="fill:url(#${id}-metal)"`)
    .replaceAll(
      'class="glass"',
      `class="glass" style="fill:url(#${id}-glass)"`,
    );
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 160 160" width="160" height="160" class="equipment-art" aria-hidden="true" ${attributes}><defs>${metal}${glass}</defs>${art}</svg>`;
}
