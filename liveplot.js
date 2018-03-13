#!/usr/bin/env node
const { spawn, exec } = require('child_process');
const readline = require('readline');
const fs = require('fs');

const EXE = process.argv[2];

if (EXE == null) {
    console.log("Usage: " + process.argv[1] + " <path_to_exe> [--watch]");
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
				run();
				queuedRun = null;
			}, 250);
		}
	});
	run();
} else {
	run(['--pause']);
}
