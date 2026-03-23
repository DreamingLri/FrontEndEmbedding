
import fs from 'fs';
const data = JSON.parse(fs.readFileSync('d:/Project/SuAsk/SuAsk_Agent/FrontEnd/public/data/frontend_metadata_dmeta_small.json', 'utf8'));
const list = Array.isArray(data) ? data : (data.data || []);
console.log('Total metadata items:', list.length);

const otids = new Set(list.map(i => i.parent_otid || i.parent_pkid || i.id));
console.log('Unique parent IDs:', otids.size);

const typeCounts = {};
list.forEach(i => {
    typeCounts[i.type] = (typeCounts[i.type] || 0) + 1;
});
console.log('Type counts:', typeCounts);
console.log('Sample item:', JSON.stringify(list[0], null, 2));
