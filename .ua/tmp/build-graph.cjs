const fs = require('fs');
const path = require('path');
const root = process.cwd();
const ua = path.join(root, '.ua');
const scan = JSON.parse(fs.readFileSync(path.join(ua, 'intermediate', 'scan-result.json')));
const typeFor = f => f.fileCategory === 'docs' ? 'document' : f.fileCategory === 'config' ? 'config' : f.fileCategory === 'data' ? 'table' : f.fileCategory === 'infra' ? 'resource' : 'file';
const idFor = f => `${typeFor(f)}:${f.path.replaceAll('\\','/')}`;
const nodes = scan.files.filter(f => !f.path.startsWith('.ua/')).map(f => ({
  id: idFor(f), type: typeFor(f), name: path.basename(f.path), filePath: f.path,
  summary: `${f.fileCategory} file in the demand forecasting and inventory planning project.`,
  tags: [f.fileCategory, f.language]
}));
const ids = new Set(nodes.map(n => n.id));
const edges = [];
const py = scan.files.filter(f => f.language === 'python' && !f.path.startsWith('.ua/'));
for (const f of py) {
  const text = fs.readFileSync(path.join(root, f.path), 'utf8');
  for (const m of text.matchAll(/(?:from|import)\s+([A-Za-z_][\w.]*)/g)) {
    const top = m[1].split('.')[0];
    const target = py.find(x => x.path === `${top}.py` || x.path === `${top}/__init__.py` || x.path === `demand_forecast/${top}.py` || x.path === `demand_forecast/${top}/__init__.py`);
    if (target && target.path !== f.path) edges.push({source:idFor(f), target:idFor(target), type:'imports', weight:0.7});
  }
}
const layers = [
  {id:'layer:documentation-and-configuration', name:'Documentation and Configuration', description:'Project guidance, metadata, issue templates, and dependency configuration.', nodeIds:nodes.filter(n=>['document','config'].includes(n.type)).map(n=>n.id)},
  {id:'layer:forecasting-library', name:'Forecasting Library', description:'Reusable forecasting, metrics, model, evaluation, inventory, and hierarchy modules.', nodeIds:nodes.filter(n=>n.filePath.startsWith('demand_forecast/') || n.filePath === 'main.py').map(n=>n.id)},
  {id:'layer:experiments-and-validation', name:'Experiments and Validation', description:'Notebook workflows, system checks, and recorded result tables.', nodeIds:nodes.filter(n=>n.filePath.startsWith('notebooks/') || n.filePath.startsWith('scripts/') || n.type === 'table').map(n=>n.id)}
];
const graph = {version:'1.0.0', project:{name:'demand-forecasting-for-inventory-planning', languages:['python','markdown','yaml','toml','csv','jupyter-notebook'], frameworks:['pandas','scikit-learn','uv'], description:'Demand forecasting experiments and reusable pipelines for inventory planning.', analyzedAt:new Date().toISOString(), gitCommitHash:require('child_process').execSync('git rev-parse HEAD',{encoding:'utf8'}).trim()}, nodes, edges, layers, tour:[{order:1,title:'Project Overview',description:'Read the README and project configuration to understand the forecasting scope.',nodeIds:nodes.filter(n=>n.id==='document:README.md'||n.id==='config:pyproject.toml').map(n=>n.id)},{order:2,title:'Reusable Forecasting Modules',description:'Explore the demand_forecast package and its advanced inventory-planning components.',nodeIds:nodes.filter(n=>n.filePath.startsWith('demand_forecast/')).slice(0,8).map(n=>n.id)},{order:3,title:'Experiments and Results',description:'Follow the notebooks, checks, and recorded result tables used to evaluate forecasts.',nodeIds:nodes.filter(n=>n.filePath.startsWith('notebooks/')||n.filePath.startsWith('data/results/')).slice(0,12).map(n=>n.id)}]};
fs.writeFileSync(path.join(ua,'intermediate','assembled-graph.json'), JSON.stringify(graph,null,2));
fs.writeFileSync(path.join(ua,'knowledge-graph.json'), JSON.stringify(graph,null,2));
console.log(JSON.stringify({nodes:nodes.length,edges:edges.length,layers:layers.length,tour:graph.tour.length},null,2));
