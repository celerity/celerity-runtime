#!/usr/bin/env node
const { spawn, exec } = require('child_process');
const readline = require('readline');
const fs = require('fs');

// Assuming windows build for now
const EXE = './build/Debug/full_stack_example.exe';

function run(args) {
	const child = spawn(
		EXE,
		args,
		{ stdio: ['inherit', 'pipe', 'inherit'] }
	);

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

if (process.argv[2] === '--watch') {
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
