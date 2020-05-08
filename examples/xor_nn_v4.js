let lossP;
let nn;

let resolution = 25;
let cols;
let rows;

let inputs = [];

let avg_time = 0;

const training_data = [{
		"input": [0, 0],
		"target": [0]
	},
	{
		"input": [1, 0],
		"target": [1]
	},
	{
		"input": [0, 1],
		"target": [1]
	},
	{
		"input": [1, 1],
		"target": [0]
	},
	// {"input": [.5, .5], "target": [.7]}

];

const test_data = [{
		"input": [0, 0],
		"target": [0]
	},
	{
		"input": [1, 0],
		"target": [1]
	},
	{
		"input": [0, 1],
		"target": [1]
	},
	{
		"input": [1, 1],
		"target": [0]
	}
];

function setup() {
	createCanvas(400, 600);
	cols = width / resolution;
	rows = width / resolution;

	// Create the input data

	for (let i = 0; i < cols; i++) {
		for (let j = 0; j < rows; j++) {
			let x1 = i / cols;
			let x2 = j / rows;
			inputs.push([x1, x2]);
		}
	}

	let nn_definition = {
		learning_rate: 0.1,
		method: "regression",
		layers: [{
				type: "dense",
				input: 2,
				nodes: 5,
				activation: "sigmoid"
			},


			{
				type: "output",
				nodes: 1,
				activation: "sigmoid"
			}
		]
	};

	nn = new NeuralNetwork(nn_definition);

	nn.loadData(training_data, test_data);

	lossP = createP();
}

function draw() {


	background(0);
	// console.log(nn);
	nn.train(1);
	lossP.html("MSE: " + nf(nn.loss, 0, 6));


	let index = 0;
	for (let i = 0; i < cols; i++) {
		for (let j = 0; j < rows; j++) {
			let ys = nn.predict(inputs[index]);
			let br = ys[0] * 255
			stroke(0);
			fill(br);
			rect(i * resolution, j * resolution, resolution, resolution);
			fill(255 - br);
			noStroke();
			textSize(8);
			textAlign(CENTER, CENTER);
			text(nf(ys[0], 1, 2), i * resolution + resolution / 2, j * resolution + resolution / 2)
			index++;
		}
	}

	lossP.html(lossP.html() + "<br/> epochs so far: " + nn.history.loss_per_epoch.length);
	//lossP.html(lossP.html() + "<br/> avg_time: " + nf(avg_time, 0, 3) + ";" + nf((avg_time * 1000) / nn.history.loss_per_epoch.length, 0, 3));

	stroke(255);
	strokeWeight(1);
	noFill();
	translate(0, 400);
	beginShape();
	let maxx = nn.history.loss_per_epoch.length;
	for (let x = 0; x < maxx; x++) {
		vertex(map(x, 0, maxx - 1, 0, width - 1), map(nn.history.loss_per_epoch[x], .4, 0, 0, 200));
	}
	endShape();
	// noLoop();
}