#!/usr/bin/env node
const { spawn, exec } = require('child_process');
const readline = require('readline');
const fs = require('fs');

// Assuming windows builder for now
const child = spawn('./build/Debug/full_stack_example.exe', ['--pause'], { stdio: ['inherit', 'pipe', 'inherit'] });

const rl = readline.createInterface({
  input: child.stdout
});

let currentGraphData = '';

function processLine(line) {
  if (!line.startsWith('#G:')) {
    console.log(line);
    return;
  }

  const matches = /#G:(\w+)#(.*)/.exec(line);
  if (matches === null) {
    throw new Error('Invalid graph line: ' + line);
  }

  const name = matches[1];
  const data = matches[2];

  if (data == '') {
    plotGraph(name);
    currentGraphData = '';
  } else {
    currentGraphData += data;
  }
}

function plotGraph(graphName) {
  console.log('Plotting ' + graphName);
  const ws = fs.createWriteStream('node_' + graphName + '.png');
  const dot = spawn('dot', ['-Tpng:gd']);
  dot.stdout.pipe(ws);
  dot.stdin.write(currentGraphData);
  dot.stdin.end();

  dot.on('exit', () => { ws.end(); })
}

rl.on('line', line => {
  processLine(line);
});

