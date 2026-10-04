import { element, svgElement, formatNumber } from "./common.js";

function curveCard(plot) {
  const light=document.body.dataset.theme==="light";
  const card=element("section",{class:"fit-objective-card"});card.append(element("h4",{},plot.title));
  const pairs=plot.series.flatMap(series=>series.x.map((x,index)=>[x,series.y[index]])).filter(([x,y])=>typeof x==="number"&&typeof y==="number"&&Number.isFinite(x)&&Number.isFinite(y));
  if(!pairs.length){card.append(element("p",{class:"fit-error"},"No qualified model/data coordinates are available for this plot."));return card;}
  let xmin=Math.min(...pairs.map(([x])=>x)),xmax=Math.max(...pairs.map(([x])=>x)),ymin=Math.min(...pairs.map(([,y])=>y)),ymax=Math.max(...pairs.map(([,y])=>y));
  const xp=Math.max((xmax-xmin)*0.04,Math.max(Math.abs(xmin),1)*0.005),yp=Math.max((ymax-ymin)*0.08,Math.max(Math.abs(ymin),1)*0.005);
  xmin-=xp;xmax+=xp;ymin-=yp;ymax+=yp;
  const x=v=>65+(v-xmin)/(xmax-xmin)*620,y=v=>315-(v-ymin)/(ymax-ymin)*260;
  const svg=svgElement("svg",{viewBox:"0 0 760 385",role:"img","aria-label":`${plot.title}: ${plot.x_label} versus ${plot.y_label}`});
  for(let tick=0;tick<=5;tick++){
    const xv=xmin+(xmax-xmin)*tick/5,yv=ymin+(ymax-ymin)*tick/5;
    svg.append(svgElement("path",{d:`M${x(xv)} 55V315 M65 ${y(yv)}H685`,stroke:light?"#ccd8df":"#36505a",fill:"none"}));
    for(const [label,attrs] of [[formatNumber(xv,4),{x:x(xv),y:336,"text-anchor":"middle"}],[formatNumber(yv,4),{x:58,y:y(yv)+4,"text-anchor":"end"}]]){
      const text=svgElement("text",{...attrs,fill:"currentColor","font-size":11});text.textContent=label;svg.append(text);
    }
  }
  for(const [label,attrs] of [[plot.x_label,{x:375,y:371,"text-anchor":"middle"}],[plot.y_label,{x:65,y:22}]]){const text=svgElement("text",{...attrs,fill:"currentColor","font-size":12});text.textContent=label;svg.append(text);}
  const legend=element("div",{class:"fit-objective-legend"}),detail=element("p",{class:"field-help"},"Hover, focus or click data points to inspect their values. Hollow markers are validation-only observations.");
  plot.series.forEach(series=>{
    const color=/vapor|β|γ2|gamma2/i.test(series.name)?(light?"#8e601a":"#f3bf75"):/perfect/i.test(series.name)?(light?"#516e7c":"#8095a0"):(light?"#146f61":"#78e6cb");
    const label=element("span"),swatch=element("i");swatch.style.background=color;label.append(swatch,document.createTextNode(series.name));legend.append(label);
    if(series.mode==="line"){
      let path="",open=false;
      series.x.forEach((xv,index)=>{const yv=series.y[index];if(typeof xv!=="number"||typeof yv!=="number"||!Number.isFinite(xv)||!Number.isFinite(yv)){open=false;return;}path+=`${open?"L":"M"}${x(xv)} ${y(yv)} `;open=true;});
      svg.append(svgElement("path",{d:path,fill:"none",stroke:color,"stroke-width":2}));
    }else{
      series.x.forEach((xv,index)=>{
        const yv=series.y[index];if(typeof xv!=="number"||typeof yv!=="number"||!Number.isFinite(xv)||!Number.isFinite(yv))return;
        const circle=svgElement("circle",{cx:x(xv),cy:y(yv),r:4,fill:series.role==="validation"?"none":color,stroke:color,"stroke-width":2,tabindex:"0",role:"button","aria-label":`${series.name}, ${formatNumber(xv)}, ${formatNumber(yv)}`});
        const title=svgElement("title");title.textContent=`${series.name}: ${plot.x_label} ${formatNumber(xv)}, ${plot.y_label} ${formatNumber(yv)}`;circle.append(title);
        const inspect=()=>detail.textContent=title.textContent;circle.addEventListener("mouseenter",inspect);circle.addEventListener("focus",inspect);circle.addEventListener("click",inspect);svg.append(circle);
      });
    }
  });
  card.append(svg,legend,detail);
  if(plot.errors.length)card.append(element("p",{class:"fit-error"},`${plot.errors.length} sampled model states are unavailable; curves show gaps. ${plot.errors[0].error}`));
  return card;
}

export function renderObjectivePlots(container,plots) {
  container.replaceChildren();
  if(!plots?.length){container.append(element("p",{class:"field-help"},"Run the fit to generate physical objective curves and data/model comparisons."));return;}
  const types=[...new Set(plots.map(plot=>plot.kind))],select=element("select",{"aria-label":"Objective plot type"}),content=element("div");
  select.append(element("option",{value:"all"},"All objective plots"));for(const type of types)select.append(element("option",{value:type},type));
  const show=()=>content.replaceChildren(...plots.filter(plot=>select.value==="all"||plot.kind===select.value).map(curveCard));select.onchange=show;
  container.append(select,content);show();
}
