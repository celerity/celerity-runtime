#!/usr/bin/env node
const { spawn, exec } = require('child_process');
const readline = require('readline');
const fs = require('fs');

const EXE = process.argv[2];

if (EXE == null) {
    console.log("Usage: " + process.argv[1] + " <path_to_exe> [--watch] [-- <additional args>]");
    process.exit(0);
}

function run(args) {
  console.log(">> Running...");
  let child;
  try {
    child = spawn(
      EXE,
      args,
      { stdio: ['inherit', 'pipe', 'inherit'] }
    );
  } catch(e) {
    console.log('Failed to spawn process (' + e.code + ')');
    return;
  }

  child.on('error', (err) => {
    console.log('Error during execution');
  });

  child.on('exit', (code) => {
    console.log('Process ended (' + code + ')');
  });

  const rl = readline.createInterface({
    input: child.stdout
  });

  function processLine(line) {
    if (line === '') return;
    try {
        const parsed = JSON.parse(line);
        if (parsed.channel !== 'graph') {
            console.log(line);
            return;
        }
        plotGraph(parsed.name, parsed.data);
    } catch (e) {
        console.log(line);
        return;
    }
  }

  function plotGraph(name, data) {
    console.log('Plotting ' + name);
    const ws = fs.createWriteStream('node_' + name + '.png');
    const dot = spawn('dot', ['-Tpng:gd']);
    dot.stdout.pipe(ws);
    dot.stdin.write(data, () => {
        dot.stdin.end();
    });

    dot.stdout.on('close', () => { ws.end(); })
  }

  rl.on('line', line => {
    processLine(line);
  });
}

let additionalArgs = [];
const argsIdx = process.argv.indexOf('--');
if (argsIdx !== -1) {
    additionalArgs = process.argv.slice(argsIdx + 1);
}

if (process.argv[3] === '--watch') {
  let queuedRun = null;
  fs.watch(EXE, {}, (eventType, filename) => {
    if (eventType === 'change') {
      // Work around fs.watch firing multiple events quickly
      if (queuedRun !== null) {
        clearTimeout(queuedRun);
      }
      queuedRun = setTimeout(() => {
        run(additionalArgs);
        queuedRun = null;
      }, 250);
    }
  });
  run();
} else {
  run(additionalArgs);
}
