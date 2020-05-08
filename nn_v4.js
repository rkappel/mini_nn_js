class ActivationFunction {
	constructor(func, dfunc) {
		this.func = func;
		this.dfunc = dfunc;
	}
}

let sigmoid = new ActivationFunction(
	function (arr) {
		if (arr instanceof Matrix) {
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					arr.data[i][j] = 1 / (1 + Math.exp(-arr.data[i][j]));
				}
			}
		} else {
			for (let i = 0; i < arr.length; i++) {
				arr[i] = 1 / (1 + Math.exp(-arr[i]));
			}
		}		
		return arr;
	},
	function (arr) {
		if (arr instanceof Matrix) {
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					arr.data[i][j] = arr.data[i][j] * (1 - arr.data[i][j]);
				}
			}
		} else {
			for (let i = 0; i < arr.length; i++) {
				arr[i] = arr[i] * (1 - arr[i]);
			}
		}		
		return arr;
	}
);

let tanh = new ActivationFunction(
	// x => Math.tanh(x),
	// y => 1 - (y * y)
	function (arr) {
		if (arr instanceof Matrix) {
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					arr.data[i][j] = Math.tanh(arr.data[i][j]);
				}
			}
		} else {
			for (let i = 0; i < arr.length; i++) {
				arr[i] = Math.tanh(arr[i]);
			}
		}		
		return arr;
	},
	function (arr) {
		if (arr instanceof Matrix) {
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					arr.data[i][j] = 1 - (arr.data[i][j]**2);
				}
			}
		} else {
			for (let i = 0; i < arr.length; i++) {
				arr[i] = 1 - (arr[i]**2);
			}
		}		
		return arr;
	}
);

let ReLU = new ActivationFunction(
	// x => (x <= 0) ? 0 : 1 / (1 + Math.exp(-x)),
	// y => (y <= 0) ? 0 : y * (1 - y)
	function (arr) {
		if (arr instanceof Matrix) {
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					arr.data[i][j] = (arr.data[i][j] <= 0) ? 0 : 1 / (1 + Math.exp(-arr.data[i][j]));
				}
			}
		} else if (arr instanceof Array) {
			for (let i = 0; i < arr.length; i++) {
				arr[i] = (arr[i] <= 0) ? 0 : 1 / (1 + Math.exp(-arr[i]));
			}
		} else {
			arr = (arr <= 0) ? 0 : 1 / (1 + Math.exp(-arr));
		}
		return arr;
	},
	function (arr) {
		if (arr instanceof Matrix) {
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					arr.data[i][j] = (arr.data[i][j] <= 0) ? 0 : arr.data[i][j] = arr.data[i][j] * (1 - arr.data[i][j]);
				}
			}
		} else if (arr instanceof Array) {
			for (let i = 0; i < arr.length; i++) {
				arr[i] = (arr[i] <= 0) ? 0 : arr[i] * (1 - arr[i]);
			}
		} else {
			arr = (arr <= 0) ? 0 : arr * (1 - arr);
		}		
		return arr;
	}
);

let SoftMax = new ActivationFunction(
	function (arr) {
		if (arr instanceof Matrix) {
			let sum = 0;
			let max = -Infinity;
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					max = Math.max(arr.data[i][j], max);
				}
			}
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					arr.data[i][j] = Math.exp(arr.data[i][j] - max);
					sum += arr.data[i][j];
				}
			}
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					arr.data[i][j] /= sum;
				}
			}
		} else if (arr instanceof Array) {
			let sum = 0;
			let max = -Infinity;
			for (let i = 0; i < arr.length; i++) {
				max = Math.max(arr[i], max);
			}
			for (let i = 0; i < arr.length; i++) {
				arr[i] = Math.exp(arr[i] - max)
				sum += arr[i];
			}
			for (let i = 0; i < arr.length; i++) {
				arr[i] /= sum;
			}
		} else {
			throw(new Error("SoftMax requires a Matrix or Array"));
		}
		return arr;
	},
	function (arr) {
		if (arr instanceof Matrix) {
			for (let i = 0; i < arr.rows; i++) {
				for (let j = 0; j < arr.cols; j++) {
					arr.data[i][j] = (arr.data[i][j] <= 0) ? 0 : arr.data[i][j] * (1 - arr.data[i][j]);
				}
			}
		} else if (arr instanceof Array) {
			for (let i = 0; i < arr.length; i++) {
				arr[i] = (arr[i] <= 0) ? 0 : arr[i] * (1 - arr[i]);
			}
		} else {
			throw(new Error("SoftMax requires a Matrix or Array"));
		}		
		return arr;
	}
);


class LossFunction {
	constructor(func, dfunc) {
		this.func = func;
		this.dfunc = dfunc;
	}
}

let mse = new LossFunction(
	(guess, target) => {
	// console.log(guess);
	// console.log(target);
	
		let t = 0;
		if (guess instanceof Matrix) {
			for (let i = 0; i < guess.rows; i++) {
				for (let j = 0; j < guess.cols; j++) {
					t += (guess.data[i][j] - target.data[i][j])**2;
				}
			}
			return t/guess.rows;
		} else if (guess instanceof Array) {
			for (let n = 0; n < guess.length; n++) {
				t += (guess[n]-target[n])**2;
			}
			return t/guess.length;
		}
		
	},
	
	(guess, target) => {
		let errors = [];
		if (guess instanceof Matrix) {
			return Matrix.sub(guess,target);
		} else if (guess instanceof Array) {
			for (let n = 0; n < guess.length; n++) {
				errors.push(guess[n] - target[n]);
			}		
			return errors;
		}
	}
);

let smax = new LossFunction(
	(errors) => {
		let sum = 0;
		for (let n = 0; n < errors.length; n++) {
			errors[n] = Math.exp(errors[n]);
			sum += errors[n];
		}
		for (let n = 0; n < errors.length; n++) {
			errors[n] /= sum;
		}
		return t/errors.length;
	},
	
	(errors) => {
		return errors;
	}
);


class OptimizerFunction {
	constructor(func) {
		this.func = func;
	}
}

let sgd = new OptimizerFunction();

let adam = new OptimizerFunction();

// NeuralNetwork
//
//	Structure:
//		.layers - Array of layers containing meta_data, weights and biases
//			- output layer has information about classification (labels)
//		.data - intermediate storage of training and testing data
//		
//		.history - historic training metrics
//	Usage:
//		

class NeuralNetwork {

	constructor(layout) {

		this.definition = layout;
		
		this.history = {};
		this.history.loss_per_batch = [];
		this.history.loss_per_epoch = [];
		
		
		((layout.learning_rate === undefined) ? (this.lr = 0.01) : (this.lr = layout.learning_rate));

		this.loss_fn = mse;
		
		switch (layout.method) {
			case "regression":
			case "reg":
				this.method = "regression";
				break;
			case "categorization":
				this.method = "categorization";
				break;
			
			default: "regression";
			
		}

		this.loss = Infinity;

		this.training_cycles = 0;

		this.layers = [];
		let act;
		
		for (let i = 0; i < layout.layers.length; i++) {
			switch (layout.layers[i].activation) {
				case "ReLU":
				case "relu":
				case "RELU":
					act = ReLU;
					break;
				case "tanh":
					act = tanh;
					break;
				case "SoftMax":
				case "Softmax":
				case "softmax":
				case "smax":
					act = SoftMax;
					break;
				case "sigmoid":
				default:
					act = sigmoid;
			}
			//console.log("i: "+i+ " --> ");
			//console.log(layout.layers[i]);
			switch (layout.layers[i].type) {
				case "dense":
				case "full":
				case "fc":
					if (layout.layers[i].input === undefined) {
						this.layers.push(new nn_layer_dense(layout.layers[i].nodes, this.layers[this.layers.length-1].nodes.rows, act));
					} else {
						this.layers.push(new nn_layer_dense_w_input(layout.layers[i].nodes, layout.layers[i].input, act));
					}
					break;
				case "conv":
					if (layout.layers[i].input === undefined) {
						this.layers.push(new nn_layer_conv(layout.layers[i].nodes, this.layers[this.layers.length-1].nodes.length, [this.layers[this.layers.length-1].nodes_size_x,this.layers[this.layers.length-1].nodes_size_y], layout.layers[i].kernel_size, layout.layers[i].stride, act));
					} else {
						this.layers.push(new nn_layer_conv_w_input(layout.layers[i].nodes, layout.layers[i].input, layout.layers[i].kernel_size, layout.layers[i].stride, act));
					}
					break;
				case "maxpool":
					this.layers.push(new nn_layer_maxpool(this.layers[this.layers.length-1].nodes.length, [this.layers[this.layers.length-1].nodes_size_x, this.layers[this.layers.length-1].nodes_size_y], layout.layers[i].pool_size));
					break;
				case "serial":
				case "serialize":
					// console.log(this.layers[this.layers.length-1].nodes);
					this.layers.push(new nn_layer_serialize(this.layers[this.layers.length-1].nodes));
					break;
				case "out":
				case "output":
					this.layers.push(new nn_layer_output(layout.layers[i].nodes, this.layers[this.layers.length-1].nodes.rows, act));
					break;
			}
			
			//console.log(this.layers[i]);
		}
	}
	
	loadData(training_data, test_data) {
	
		// data is expected as array of objects
		// each object having one input and one target value {"input": input_data_array, "target":target_value"}
		// input being an array according to layer.input definition
		// output being an array of values (mode: regression) or a label (mode: categorization)
		// in case of mode: categorization the label are converted
		
		this.training_data = training_data;
		let data_shape = mda_shape(this.training_data[0].input);
		
		// if (data_shape[0] != this.layers[0].input_size_x || data_shape[1] !=  this.layers[0].input_size_y) throw("Data needs to fit definition!");
		
		let template = [];
		if (this.method == "categorization") {
			// build list of all labels used in the training set
			this.labels = [];
			for (let x = 0; x < this.training_data.length; x++) {
				this.labels.push(this.training_data[x].target);
			}		

			// console.log(this.labels);
			this.labels = this.labels.unique();
			// console.log(this.labels);
			
			for (let i = 0; i < this.labels.length; i++) template.push(0);

			for (let x = 0; x < this.training_data.length; x++) {
				this.training_data[x].label = this.training_data[x].target;
				let i = this.labels.findIndex(v => v == this.training_data[x].label);
				this.training_data[x].target = template.slice(0);
				this.training_data[x].target[i] = 1;
			}
			
		}
		
		this.test_data = test_data;
		
		// if (this.method == "categorization") {

			// for (let x = 0; x < this.test_data.length; x++) {
				// this.test_data[x].label = this.test_data[x].target;
				// let i = this.labels.findIndex(v => v == this.test_data[x].label);
				// this.test_data[x].target = template.slice(0);
				// this.test_data[x].target[i] = 1;
			// }
			
		// }
	}
	
	
	predict(input, mode) {
	
		let temp_out;
		let temp_in;
		let output = [];
		
		(mda_shape(input).length == 1) ? temp_in = new Matrix(input) : temp_in = [input];
		
		for (let i = 0; i < this.layers.length; i++) {
			// console.log(temp_in);
			temp_out = this.layers[i].feedforward(temp_in);
			temp_in = temp_out;
		}
			
		if (this.method == "regression") {
			return (mode == "M") ? temp_out : temp_out.toArray();
		} else {
			temp_out = temp_out.toArray();
			for (let i = 0; i < temp_out.length; i++) {
				output.push({"label": this.labels[i], "value": temp_out[i]});
			}
			output.sort((a,b) => (b.value - a.value));
			return output;
		}
	}

	train(epochs = 1, batch_size = 1) {
	
		let method_backup = this.method;
		this.method = "regression";
		let t0 = performance.now();
		let epoch_loss;
		for (let e = 0; e < epochs; e++) {
			epoch_loss = 0;
			// prepare batches
			this.training_data.shuffle();
		
			let last_batch = this.training_data.length % batch_size;
			let batches = floor(this.training_data.length / batch_size);
			
			if (last_batch == 0) {
				last_batch = batch_size;
				batches--;
			}
						
			// console.log("this.training_data.length: " + this.training_data.length);
			// console.log("batch_size: " + batch_size);
			// console.log("batches: " + batches);
			// console.log("last_batch: " + last_batch);
			
			let batch_loss;
			for (let b = 0; b <= batches; b++) {
				batch_loss = 0;
				// console.log("b: " + b);
				let max_s;

				if (b >= batches) {
					max_s = last_batch;
				} else {
					max_s = batch_size;
				}
				// console.log("max_s: " + max_s);
				
				let errors = [];
				
				let guess, target;
				for (let s = 0; s < max_s; s++) {
					let idx = b * batch_size + s;
					guess = this.predict(this.training_data[idx].input,"M");
					// console.log("guess");
					// console.log(guess);
					
					target = new Matrix(this.training_data[idx].target);
					// console.log("target");
					// console.log(target);
					
					errors.push(this.loss_fn.dfunc(guess, target));
				}
				
				for (let i = 0; i < errors.length; i++) {
					// calculate Loss
					let single_loss = this.loss_fn.func(guess, target);
					epoch_loss += single_loss;
					batch_loss += single_loss;
				
					
					// backpropagate 
					let temp_in = errors[i];	
					
					for (let i = this.layers.length - 1; i > 0; i--) {
						temp_in = this.layers[i].backpropagate(this.layers[i-1].nodes,temp_in,this.lr);
						}
					this.layers[0].backpropagate(temp_in, this.lr);
				}
				this.history.loss_per_batch.push(batch_loss/batch_size);
			
			}
			// console.log(count);
			this.method = method_backup;
			
			this.training_cycles++;
		}
		this.loss = epoch_loss/this.training_data.length;
		this.history.loss_per_epoch.push(this.loss);
		let t1 = performance.now();
		
		avg_time += (t1 - t0)/this.training_data.length;
		// console.log(nf((t1 - t0)/this.training_data.length,0,2) + " milliseconds / input");
		
		return true;
	
	}

	test(sample_size) {
	
		if (this.method == "regression") return false;
		
		let correct = 0;
		if (sample_size === undefined) sample_size = this.test_data.length;
		
		for (let k = 0; k < sample_size; k++) {
			let idx = floor(random(this.test_data.length));
		
			let guess = this.predict(this.test_data[idx].input);

			(guess[0].label == this.test_data[idx].target) ? correct++ : false ;
			
		}
		return (correct / sample_size);
	}
	
	save() {
	
		// let json_data = JSON.stringify(this);
		// console.log(json_data);
		saveJSON(this,'mynet.nn');
		
	}
	
	loadConfig(data) {
	
		console.log(data);
		
		this.loss_fn = mse;		
		this.loss = data.loss;
		
		this.lr = data.lr;
		
		this.method = data.method;
		this.training_cycles = data.training_cycles;
		this.training_data = data.training_data;
		this.test_data = data.test_data;
		this.labels = data.labels;
		this.history = data.history;
		
		for (let l = 0; l < data.layers.length; l++) {
			switch (data.layers[l].type) {
				case "input_dense":
					// this.layers[l].input.data = data.layers[l].input.data;
				case "dense":
					// this.layers[l].nodes.data = data.layers[l].nodes.data;
					// this.layers[l].errors.data = data.layers[l].errors.data;
					this.layers[l].bias.data = data.layers[l].bias.data;
					this.layers[l].weights.data = data.layers[l].weights.data;
					break;
				case "conv":
					this.layers[l].bias = data.layers[l].bias;
					this.layers[l].kernels = data.layers[l].kernels;
					break;
				case "input_conv":
					this.layers[l].bias = data.layers[l].bias;
					this.layers[l].kernels = data.layers[l].kernels;
					break;
				case "maxpool":
					// this.layers[l].nodes.data = data.layers[l].nodes.data;
					// this.layers[l].errors.data = data.layers[l].errors.data;
					break;
				case "pool":
					// this.layers[l].nodes.data = data.layers[l].nodes.data;
					// this.layers[l].errors.data = data.layers[l].errors.data;
					break;
				case "serial":
					// this.layers[l].nodes.data = data.layers[l].nodes.data;
					// this.layers[l].errors.data = data.layers[l].errors.data;
					break;
				case "output":
					// this.layers[l].nodes.data = data.layers[l].nodes.data;
					// this.layers[l].errors.data = data.layers[l].errors.data;
					this.layers[l].bias.data = data.layers[l].bias.data;
					this.layers[l].weights.data = data.layers[l].weights.data;
					break;
			}
			
		}

			
	}
	
}

// each layer manages it's processes on it'S own
// only the interfacing to other layers is standardized
// each layer needs to provide:
// constructor
// feed_forward
//   input: (as a Matrix) and output (as a Matrix) to be uased a input for the following layer
//   output:
// backpropagate
//   input (as a Matrix) and output (as a Matrix) to be used in the previous layer
//   output
// adjust
//   input
//   ouput
// creates nodes, error, bias and weights Matrices
// a feedforward function with input (as a Matrix) and output (as a Matrix) to be uased a input for the following layer
// a calculate error function with input (as a Matrix) and output (as a Matrix) to be used in the previous layer
// an adjust function that calculates the adjustment of the layer's weights and biases using the previous layer's information

class nn_layer_dense {
	
	constructor(nodes, input_nodes, activation_function) {
		this.type = "dense";
		
		this.nodes = new Matrix(nodes,1);
		this.errors = new Matrix(nodes,1);
		this.bias = new Matrix(nodes,1);  //bias is initialized with zeros
		
		(activation_function === undefined) ? this.act_fn = sigmoid : this.act_fn = activation_function;
		
		// initialize  weights
		let init_range = Math.sqrt(6 / (input_nodes + nodes));
		
		this.weights = new Matrix(nodes, input_nodes,"rand+-1");
		this.weights.map(x => x*init_range);
		
	}
	
	feedforward(input) {
	// Input: Matrix with x rows and 1 col (x defined by number of neurons in the prev layer
	// Output: Matrix with neurons of this layer

		this.nodes = Matrix.dot(this.weights, input);
		// console.log("after dot-product with weights");
		// this.nodes.print();
		
		this.nodes.add(this.bias);
		// console.log("after bias");
		// this.nodes.print();
		
		this.act_fn.func(this.nodes);
		// this.nodes.map(this.act_fn.func);
		// console.log("after act_fn");
		// this.nodes.print();
		
		return this.nodes
		}
		
	backpropagate(input_nodes, error, lr) {
	
	// IN: prev_nodes as Nx1 Matrix
	// OUT: error for prev layer as Nx1 Matrix

		this.errors = error;
		let gradient = this.act_fn.dfunc(this.nodes);
		gradient.mult(this.errors);
				
		let weights_transposed = Matrix.trans(this.weights);		
		//let dx = Matrix.dot(weights_transposed, gradient);
		let dx = Matrix.dot(weights_transposed, this.errors);
		
		gradient.mult(lr);
		
		this.bias.sub(gradient); // db = gradient

		let input_nodes_T = Matrix.trans(input_nodes);
		let delta_weights = Matrix.dot(gradient, input_nodes_T);
		
		this.weights.sub(delta_weights);

		return dx;
	}

}

class nn_layer_dense_w_input extends nn_layer_dense{
	
	constructor(nodes, input_nodes, activation_function) {
		super(nodes, input_nodes, activation_function);
		this.type = "input_dense";
		this.input = new Matrix(input_nodes,1);
	}
	
	feedforward(input) {
	
		// save input for future reference
		this.input = input;
		return super.feedforward(input);
	}
	
	backpropagate(derror, lr) {
		super.backpropagate(this.input, derror, lr); 
	}	
}

class nn_layer_output extends nn_layer_dense{
	
	constructor(nodes, input_nodes, activation_function) {
		super(nodes, input_nodes, activation_function)
		
		this.target = new Matrix(nodes,1);
		this.type = "output";
	}
	
	backpropagate(input_nodes, error, lr) {
		//this.target = target;  // In case I need to store the target values ...
		this.errors = error;
		return super.backpropagate(input_nodes, this.errors, lr);
	}
}


class nn_layer_conv {
	
	constructor(nodes, input_nodes, input_shape, kernel_size, stride, activation_function) {
	// The input consists of N data points, each with C channels, height H and
    // width W. We convolve each input with F different filters, where each filter
    // spans all C channels and has height HH and width WW.
    // Input:
    // - x: Input data of shape (N, C, H, W)
    // - w: Filter weights of shape (F, C, HH, WW)
    // - b: Biases, of shape (F,)
    // - conv_param: A dictionary with the following keys:
      // - 'stride': The number of pixels between adjacent receptive fields in the
        // horizontal and vertical directions.
      // - 'pad': The number of pixels that will be used to zero-pad the input. 
        
    // During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    // along the height and width axes of the input. Be careful not to modfiy the original
    // input x directly.
    // Returns a tuple of:
    // - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      // H' = 1 + (H + 2 * pad - HH) / stride
      // W' = 1 + (W + 2 * pad - WW) / stride
    // - cache: (x, w, b, conv_param)
		
		this.type = "conv";
		
		this.input_size_x = input_shape[0];
		this.input_size_y = input_shape[1];
		// this.input_size_c = input_shape[2]; // Future enhancement: Channels
		
		this.input_nodes = input_nodes;
		
		this.kernel_size_x = kernel_size;
		this.kernel_size_y = kernel_size;
		
		this.stride = stride;
		
		this.nodes_size_x = (this.input_size_x - kernel_size) / this.stride + 1;
		this.nodes_size_y = (this.input_size_y - kernel_size) / this.stride + 1;
		
		this.nodes = mda(nodes, this.nodes_size_x, this.nodes_size_y);
		this.errors = mda(nodes, this.nodes_size_x, this.nodes_size_y);
		
		let init_range = Math.sqrt(6 / ((input_nodes + nodes) * (kernel_size**2)));
		
		// creating and initializing bias
		// console.log("nodes: "+nodes);
		this.bias = mda(nodes);
		// console.log(this.bias);
		for (let i = 0; i < this.bias.length; i++) {
			this.bias[i] = 0;
		}
		
		// creating and initializing kernels
		// console.log(nodes);
		this.kernels = mda(nodes, this.input_nodes, this.kernel_size_x, this.kernel_size_y);
		
		// console.log(this.kernels);
		for (let f = 0; f < nodes; f++) {	
			for (let n = 0; n < this.input_nodes; n++) {
				for (let i = 0; i < this.kernel_size_x; i++) {
					for (let j = 0; j < this.kernel_size_y; j++) {
						this.kernels[f][n][i][j] = (random(2) - 1) * init_range;
					}
				}
			}
		}
		
		this.dw = this.kernels.slice(0);

		(activation_function === undefined) ? this.act_fn = ReLU : this.act_fn = activation_function;
	}
	
	feedforward(_input) {

		for (let f = 0; f < this.nodes.length; f++) {
			for (let i = 0; i < this.nodes_size_x; i++) {
				for (let j = 0; j < this.nodes_size_y; j++) {
					this.nodes[f][i][j] = 0;
				}
			}
		}
		// console.log("_input");
		// console.log(mda_shape(_input));
		// console.log(_input);
		for (let f = 0; f < this.nodes.length; f++) {   
		// console.table(this.nodes[f]);// kernels
			for (let n = 0; n < this.input_nodes; n++) {                             // Go throug all prev layer images
				for (let i = 0; i < this.nodes_size_x; i += this.stride) {           // ouput nodes x
					for (let j = 0; j < this.nodes_size_y; j += this.stride) {       // ouput nodes y
						// console.table(this.nodes[f]);
						for (let k = 0; k < this.kernel_size_x; k++) {
							for (let l = 0; l < this.kernel_size_y; l++) {
								// console.log(_input);
								// console.log(this.kernels);
								// console.log("f: "+f+";n: "+n+"; i: "+i+"; j: "+j+"; k: "+k+"; l: "+l);
								// console.log("_input[n][this.stride*i+k][this.stride*j+l] : "+_input[n][this.stride*i+k][this.stride*j+l]);
								// console.log("this.kernels[f][n][k][l] : "+ this.kernels[f][n][k][l]);
								this.nodes[f][i][j] += _input[n][this.stride*i+k][this.stride*j+l] * this.kernels[f][n][k][l];
							}
						}
						// console.table(this.nodes[f]);
						this.nodes[f][i][j] += this.bias[f];
						this.nodes[f][i][j]  = this.act_fn.func(this.nodes[f][i][j]);
					}
				}
			}
			// console.table(this.nodes[f]);
		}
		return this.nodes.slice(0); 
	}
	
	backpropagate(input_nodes, error, lr) {
		
		// console.log("backpropagate layer: " + this.type);
		//TODO: implement the same for multiple conv layers
		
		//let dx = mda_copy(input_nodes);
		let dx = mda(mda_shape(input_nodes));
		// console.log("input_nodes");
		// console.log(input_nodes);
		
		this.errors = mda_copy(error);
		// console.log("error");
		// console.log(this.errors);
		
		
		this.gradient = mda(this.nodes.length, this.nodes_size_x, this.nodes_size_y);
			
		// calculate gradient = error * derivative of output
		for (let f = 0; f < this.nodes.length; f++) {
			for (let i = 0; i < this.nodes_size_x; i++) {
				for (let j = 0; j < this.nodes_size_y; j++) {
					this.gradient[f][i][j] = this.act_fn.dfunc(this.nodes[f][i][j]) * this.errors[f][i][j];
				}
			}
		}
		
		// 0-padding of errors
		//let ep = mda_copy(this.errors);
		let ep = error;
	
			
		for (let f = 0; f < this.errors.length; f++) {
			for (let j = 0; j < this.nodes_size_y; j++) { 
				for (let i = 0; i < this.kernel_size_x-1; i++) { 
					ep[f][j].unshift(0);
					ep[f][j].push(0);
				}
			}
		}
			
		let t = [];
		for (let i = 0; i < this.nodes_size_x + 2 * (this.kernel_size_x-1) ; i++) t.push(0);
							
		for (let f = 0; f < this.errors.length; f++) {
			for (let x = 0; x < this.kernel_size_y-1; x++) { 
				ep[f].unshift(t);
				ep[f].push(t);
			}
		}
		
		//let w_ = mda_copy(this.kernels);
		let w_ = mda(mda_shape(this.kernels));
		
		// console.log(w_);
		for (let f = 0; f < this.nodes.length; f++) {
			for (let n = 0; n < input_nodes.length; n++) {
				for (let k = 0; k < this.kernel_size_x; k++) { 
					for (let l = 0; l < this.kernel_size_y; l++) {
						w_[f][n][k][l] = this.kernels[f][n][this.kernel_size_x - k - 1][this.kernel_size_y - l - 1];
					}
				}
			}
		}
		// calculate dx = weights x errors
		
		for (let n = 0; n < input_nodes.length; n++) {
			for (let f = 0; f < this.nodes.length; f++) {
				// console.log("shape of dx[n]: " + mda_shape(dx[n]));
				// console.log("shape of k[f][n]: " + mda_shape(this.kernels[f][n]));
				// console.log("shape of ep[n]: " + mda_shape(ep[n]));
				// console.table(ep[f]);
				for (let i = 0; i < this.input_size_x; i++) {  
					for (let j = 0; j < this.input_size_y; j++) {  
						dx[n][i][j] = 0;
						for (let k = 0; k < this.kernel_size_x; k++) { 
							for (let l = 0; l < this.kernel_size_y; l++) { 
								dx[n][i][j] += w_[f][n][k][l] * ep[f][i + k][j + l];
							}
						}
					}
				}
			}
		}
		// console.log("dx");
		// console.log(dx);
		
		
		// calculate and apply db = gradient
		// TODO: simplify db from array to re-used variable
		let db = mda(this.nodes.length);
		for (let f = 0; f < this.nodes.length; f++) {
			db[f] = 0;
			for (let i = 0; i < this.nodes_size_x; i++) {
				for (let j = 0; j < this.nodes_size_y; j++) {
					db[f] += this.gradient[f][i][j];
				}
			}
			this.bias[f] -= db[f] * lr;
		}
		
		
		// calculate and apply dw = gradient x input
		this.dw = mda(this.nodes.length, this.input_nodes, this.kernel_size_x, this.kernel_size_y);

		for (let n = 0; n < input_nodes.length; n++) {
			for (let f = 0; f < this.nodes.length; f++) {
				for (let i = 0; i < this.kernel_size_x; i++) {  // kernel rows
					for (let j = 0; j < this.kernel_size_y; j++) {  // kernel cols
						this.dw[f][n][i][j] = 0;
						for (let k = 0; k < this.nodes_size_x; k++) { //conv rows
							for (let l = 0; l < this.nodes_size_y; l++) { // conv cols
								this.dw[f][n][i][j] += this.gradient[f][k][l] * input_nodes[n][this.stride*i+k][this.stride*j+l];
							}
						}
						// console.log("dw[f][n][i][j] : "+dw[f][n][i][j]);
					}
				}
				for (let i = 0; i < this.kernel_size_x; i++) {  // kernel rows
					for (let j = 0; j < this.kernel_size_y; j++) {  // kernel cols
						this.kernels[f][n][i][j] -= this.dw[f][n][i][j] * lr;
					}
				}
				
			}
		}
			
		return dx;
	}
}

class nn_layer_conv_w_input extends nn_layer_conv {
	
	constructor(nodes, input, kernel_size, stride, activation_function) {
		
		super(nodes, 1, input, kernel_size, stride, activation_function);
		this.type = "input_conv";
	
		this.input = mda(1, this.input_size_x, this.input_size_y);
	}
	
	feedforward(input) {
	
		this.input = mda_copy(input);
		return super.feedforward(this.input);
	}

	backpropagate(error, lr) {
		super.backpropagate(this.input, error, lr);
	}
	
}

class nn_layer_maxpool {
	
	constructor(nodes, input_shape, pool_size) {
		this.type = "maxpool";
		
		this.p_size = pool_size;
		
		this.input_size_x = input_shape[0];
		this.input_size_y = input_shape[1];
		
		this.nodes_size_x = floor((this.input_size_x / this.p_size) + 0.5);
		this.nodes_size_y = floor((this.input_size_y / this.p_size) + 0.5);
		// console.log(input_shape);
		this.nodes = mda(nodes, this.nodes_size_x, this.nodes_size_y);
		this.errors = mda(nodes, this.nodes_size_x, this.nodes_size_y);
	}
	
	feedforward(_input) {
	
		for (let n = 0; n < _input.length; n++) {
			for (let i = 0; i < this.nodes_size_x; i++) {
				for (let j = 0; j < this.nodes_size_y; j++) {
					let record = _input[n][i * this.p_size][j * this.p_size];
					for (let k = 0; k < this.p_size; k++) {
						for (let l = 0; l < this.p_size; l++) {
							if (record < _input[n][i * this.p_size + k][j * this.p_size + l]) {
							record = _input[n][i * this.p_size + k][j * this.p_size + l];
							}
						}
					}
					this.nodes[n][i][j] = record;
				}
			}
		}
		return this.nodes;
	}
	
	backpropagate(input_nodes, error, lr) {
		// console.log("backpropagate layer: " + this.type);
		// TODO
		// get conv_layer error from pooling layer error
		// console.log("next_layer.p_stride: "+next_layer.p_stride);
		
		this.errors = error;
		
		let dx = mda(this.errors.length, this.input_size_x, this.input_size_y);
		// console.log(input_nodes);
		// console.log(dx);
		for (let f = 0; f < this.errors.length; f++) {
			for (let k = 0; k < this.nodes_size_x; k++) { // walk through pool rows
				for (let l = 0; l < this.nodes_size_y; l++) { // walk through pool cols
					for (let i = k * this.p_size; i < k * this.p_size + this.p_size; i++) { // walk through prev. layer (conv)
						for (let j = l * this.p_size; j < l * this.p_size + this.p_size; j++) {
							// console.log("f: "+f+"; k: "+k+"; l: "+l+"; i: "+i+"; j: "+j);
							(Math.round(input_nodes[f][i][j],7) == Math.round(this.nodes[f][k][l],7)) ? dx[f][i][j] = this.errors[f][k][l] : dx[f][i][j] = 0;
						}
					}
				}
			}
		}
		return dx;
	}
}
	
class nn_layer_serialize {
	
	constructor(input_layer) { // Matrix with n number of elements, each with rows*cols
		// console.log(input_layer);
		this.type = "serial";
		
		this.input_size_x = input_layer[0].length;
		this.input_size_y = input_layer[0][0].length;
		
		this.num_nodes = input_layer.length * this.input_size_x * this.input_size_y;
		
		this.nodes = new Matrix(this.num_nodes,1);
		this.errors = new Matrix(this.num_nodes,1);

	}
	
	feedforward (_input) {
		
		// console.log(_input);
		for (let k = 0; k < _input.length; k++) {
			(k == 0) ? this.nodes = new Matrix(_input[k]) : this.nodes = Matrix.concat(this.nodes,new Matrix(_input[k]));
		// console.log(this.nodes);
		}
		this.nodes.reshape(this.num_nodes,1);
		return this.nodes;
	}	
	
	backpropagate(input_nodes, error, lr) {
	
	// IN: upstream data as Nx1 Matrix
	// OUT: error for prev layer as [n][x][y] array
	let dx = [];
	this.errors = error;
	
	let temp = Matrix.split(this.errors, (this.input_size_x * this.input_size_y));
	
	for (let x = 0; x < temp.rows; x++) {
		// temp.data[x][0].print();
		temp.data[x][0].reshape(this.input_size_y,this.input_size_x);
		// temp.data[x][0].print();
		// console.log(temp.data[x][0].toArray());
		dx.push(temp.data[x][0].toArray());
		// console.log("OK");
		// this.errors.data[x][0].print();
	}
	// console.log(dx);
	return dx;
	}
	
	
}


// General idea of backpropagation:
// all here belong to the current layer
// calculate the gradient 
// gradient = <error of output> * <derivative of output>
// 
// delta for weights (or kernels)
// dw = gradient * input
//
// delta for bias
// db = gradient
//
// error of input (= error of output of prev. layer)
// dx = weights * error
//
// 1. take error of output (param#1) and calculate gradient
// 2. calculate error if prev. layer (dx)
// 3. calculate and apply db
// 4. with gradient and input (param#2) and calculate dw
// 5. apply dw
// 6. return dx




// END-OF-FILE