const fs = require('fs');
const path = require('path');

const filesToCopy = [
  { from: 'src/popup.html', to: 'dist/popup.html' },
  { from: 'manifest.json', to: 'dist/manifest.json' }
];

if (!fs.existsSync('dist')) fs.mkdirSync('dist');

filesToCopy.forEach(file => {
  fs.copyFileSync(path.resolve(file.from), path.resolve(file.to));
});
