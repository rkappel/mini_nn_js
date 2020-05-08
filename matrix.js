function my_round(value, decimals) {
  if (decimals === undefined) decimals = 13;
  // console.log(value);
  let result = parseFloat(value.toFixed(decimals));//Number(Math.round(value + 'e' + decimals) + 'e-' + decimals);
  // console.log(result);
  return result;
}

class Matrix {

	constructor(_rows, _cols, type) {
		let n;
		if (type === undefined) type = "zero";

		switch (true) {
			case (typeof(_rows) == "number"):
				this.rows = _rows;
				this.cols = _cols;
				this.data = [];

				for (let i = 0; i < this.rows; i++) {
					// let temp = []; //this.data[i] = [];
					this.data[i] = [];
					for (let j = 0; j < this.cols; j++) {
						switch (type) {
							case "zero":
							case 0:
							  n = 0;
							  break;
							case "ones":
							case "one":
							case 1:
							  n = 1;
							  break;
							case "unit": 
							  if (i==j) {
								n = 1;
							  } else {
								n = 0;
							  }
							  break;
							case "rand01":
							  n = random();
							  break;
							case "rand+-1":
							  n = random() * 2 - 1;
							  break;
							case "rand10":
							  n = floor(random(10));
							  break;
						}
						//temp.push(n); //this.data[i][j] = n; //floor(random(10)); //0;
						this.data[i][j] = n;
					}
					// this.data.push(temp);
				}
				this.rows = int(this.rows);
				this.cols = int(this.cols);
				break;
			
			case (_rows instanceof p5.Image): 
				this.rows = _rows.height;
				this.cols = _rows.width;
				_rows.loadPixels();
				this.data = [];
				for (let i = 0; i < this.rows; i++) {
					this.data[i] = [];
					for (let j = 0; j < this.cols; j++) {
						let idx = 4 * (i * this.cols + j);
						this.data[i][j] = (_rows.pixels[idx + 0] + _rows.pixels[idx + 1] + _rows.pixels[idx + 2]) / 3.0; 
						// console.log(_rows.pixels[idx + 0] + " " + this.data[i][j]);
					}
				}
				_rows.updatePixels();
				
				break;
				
			case (_rows instanceof Matrix): 
				this.rows = _rows.rows;
				this.cols = _rows.cols;
				this.data = mda_copy(_rows.data);
				break;
				
			case (_rows instanceof Array):
				if (_rows[0] instanceof Array) {
					this.rows = _rows.length;
					this.cols = _rows[0].length;
					this.data = _rows;
				} else {
					this.rows = _rows.length;
					this.cols = 1;
					this.data = [];
					for (let i = 0; i < this.rows; i++) {
					  // this.data[i].push(_rows[i].slice(0));
					  this.data.push([_rows[i]]);
					}
				}
				break;
		}
	}

  static add(m1, m2) {
    let result = new Matrix(m1.rows, m1.cols);
    if (m2 instanceof Matrix) {
      for (let i = 0; i < result.rows; i++) {
        for (let j = 0; j < result.cols; j++) {
          result.data[i][j] = m1.data[i][j] + m2.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < result.rows; i++) {
        for (let j = 0; j < result.cols; j++) {
          result.data[i][j] = m1.data[i][j] + m2;
        }
      }
    }
    return result;
  }

  add(m2) {
    if (m2 instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
			//console.log("i: "+i+"; j: "+j);
          this.data[i][j] += m2.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] += m2;
        }
      }
    }
  }


  static sub(m1, m2) {
	if (m1.cols != m2.cols || m1.rows != m2.rows) {
      throw (new Error("Matrix Subtraction Error! Matrices need to have the same rows and columns."));
    }
    let result = new Matrix(m1.rows, m1.cols);
    if (m2 instanceof Matrix) {
      for (let i = 0; i < result.rows; i++) {
        for (let j = 0; j < result.cols; j++) {
          result.data[i][j] = m1.data[i][j] - m2.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < result.rows; i++) {
        for (let j = 0; j < result.cols; j++) {
          result.data[i][j] -= m2;
        }
      }
    }
    return result;
  }

  sub(m2) {
	if (this.cols != m2.cols || this.rows != m2.rows) {
      throw (new Error("Matrix Subtractin Error! Matrices need to have the same rows and columns."));
    }
    if (m2 instanceof Matrix) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] -= m2.data[i][j];
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] -= m2;
        }
      }
    }
  }


  static dot(a, b) {
	 
    let result;
    if (b instanceof Matrix) {
      // Matrix multiplication
	  // m1(m rows,n cols) X m2(i rows, j cols) = m3(m rows, j cols) with n and i need to be equal
      if (a.cols != b.rows) {
        throw (new Error("Matrix multiplication error!"));
      }
      result = new Matrix(a.rows, b.cols);
      for (let i = 0; i < result.rows; i++) {
        for (let j = 0; j < result.cols; j++) {
          for (let z = 0; z < a.cols; z++) {
            result.data[i][j] += a.data[i][z] * b.data[z][j];
          }
        }
      }
	  return result;
    } else if (b instanceof p5.Vector) {
      let temp = Matrix.toMatrix(b);
      return Matrix.dot(a, temp);
    } else if (b instanceof Array) {
      let temp = Matrix.toMatrix(b);
      return Matrix.dot(a, temp);
    } else {
      // Scalar multiplication
      result = new Matrix(a.rows, a.cols);
      for (let i = 0; i < a.rows; i++) {
        for (let j = 0; j < a.cols; j++) {
          result.data[i][j] = b * result.data[i][j];
        }
      }
    }
    return result;
  }

  dot(m) {
    if (m instanceof Matrix) {
      // Matrix multiplication
      if (this.cols != m.rows) {
        throw (new Error("Matrix multiplication error!"));
      }
      let result = new Matrix(this.rows, m.cols);
      for (let i = 0; i < result.rows; i++) {
        for (let j = 0; j < result.cols; j++) {
          for (let z = 0; z < this.cols; z++) {
            result.data[i][j] += this.data[i][z] * m.data[z][j];
          }
        }
      }
      return result;
    } else {
      throw (new Error("Matrix multiplication error! Dot-Product only possible for two Matrices."));
    }
  }


  static mult(m1,m2) {
  
	let result = new Matrix(m1.rows, m1.cols);
    if (m2 instanceof Matrix) {
      // element-wise Matrix multiplication
      if (m1.rows !== m2.rows || m1.cols !== m2.cols) {
        throw ("Matrix element-wise multiplication error!");
      }
      for (let i = 0; i < m1.rows; i++) {
        for (let j = 0; j < m1.cols; j++) {
          result.data[i][j] = m1.data[i][j] * m2.data[i][j];
        }
      }
    } else {
      // Scalar multiplication
      for (let i = 0; i < m1.rows; i++) {
        for (let j = 0; j < m1.cols; j++) {
          result.data[i][j] = m1.data[i][j] * m2;
        }
      }
    }
	return result;
  }

  mult(m) {
    if (m instanceof Matrix) {
		// element-wise Matrix multiplication
      if (this.rows !== m.rows || this.cols !== m.cols) {
        throw (new Error("Matrix element-wise multiplication error!"));
      }
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] *= m.data[i][j];
        }
      }
    } else {
      // Scalar multiplication
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          this.data[i][j] *= m;
        }
      }
    }
  }


  static trans(m1) {
    let result;
    if (m1 instanceof Matrix) {
      // Matrix multiplication
      result = new Matrix(m1.cols, m1.rows);
      for (let i = 0; i < result.rows; i++) {
        for (let j = 0; j < result.cols; j++) {
          result.data[i][j] = m1.data[j][i];
        }
      }
    } else {
      throw (new Error("This function requires a Matrix object!"));
    }
    return result;
  }

  trans() {
    let result;
    // Matrix multiplication
    result = new Matrix(this.cols, this.rows);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.data[i][j] = this.data[j][i];
      }
    }
    this.rows = result.rows;
    this.cols = result.cols;
    this.data = result.data;
  }

  fill(func) {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        this.data[i][j] = func(val, i, j);
      }
    }
  }


  static map(m, func) {
    let result = new Matrix(m.rows, m.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.data[i][j] = func(m.data[i][j], i, j);
      }
    }
    return result;
  }

  map(func) {
	let temp = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        temp.data[i][j] = func(this.data[i][j], i, j);
      }
    }
	this.data = temp.data;
  }


  static normalize(m,from,to,min,max) {
	if (from === undefined) from = 0.0;
	if (to === undefined) to = 1.0;
	
	if (min === undefined) min = Infinity;
	if (max === undefined) max = -Infinity;
	
	if (min == Infinity && max == -Infinity) {
		for (let i = 0; i < m.rows; i++) {
		  for (let j = 0; j < m.cols; j++) {
			min = Math.min(min,m.data[i][j]);
			max = Math.max(max,m.data[i][j]);
		  }
		}
	}
	
	return Matrix.map(m,(v) => {return map(v,min,max,from,to);});
  }
  
  normalize(from,to) {
	if (from === undefined) from = 0.0;
	if (to === undefined) to = 1.0;
	let min = Infinity;
	let max = -Infinity;
	for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
		min = Math.min(min,this.data[i][j]);
		max = Math.max(max,this.data[i][j]);
      }
    }
	this.map((v) => {return map(v,min,max,from,to);});
  }


  static reshape(m,new_rows, new_cols) {
	if (m.rows * m.cols != new_rows * new_cols) throw (new Error("static Re-Shape Error: New shape needs to have same number of elements as old shape!"));
	let result = new Matrix(new_rows, new_cols);
	
	let r_i = 0;
	let r_j = 0;

	for (let i = 0; i < m.rows; i++) {
		for (let j = 0; j < m.cols; j++) {
			result.data[r_i][r_j] = m.data[i][j];
			if (r_j >= new_cols-1) {
				r_j = 0;
				r_i++;
			} else {
				r_j++;
			}
			
		}
	}
	return result;
  }

  reshape(new_rows, new_cols) {
	if (this.rows * this.cols != new_rows * new_cols) throw (new Error("Re-Shape Error: New shape needs to have same number of elements as old shape!"));
	let result = new Matrix(new_rows, new_cols);
	
	let r_i = 0;
	let r_j = 0;

	for (let i = 0; i < this.rows; i++) {
		//result[r_i]=[];
		for (let j = 0; j < this.cols; j++) {
		//console.table(result);
		if (r_j > new_cols-1) {
			r_j = 0;
			r_i++;
		} 
		result.data[r_i][r_j] = this.data[i][j];
		r_j++;
		//console.table(result);
		}
	}
	
	this.data = result.data;
	this.rows = new_rows;
	this.cols = new_cols;
  }


  static concat(m1,m2) {
	if (m1.cols != m2.cols) throw(new Error("Can't concatenate due to different number of columns!"));
	let result = new Matrix(m1.rows + m2.rows,m1.cols);
	for (let i = 0; i < m1.rows; i++) {
		result.data[i] = m1.data[i];
	}
	for (let i = 0; i < m2.rows; i++) {
		result.data[m1.rows + i] = m2.data[i];
	}
	return result;
  }

/*
//This doesn't work for some reason ....
	
  concat(m) {
	if (this.cols != m.cols) throw("Can't concatenate due to different number of columns!");
	for (let i = 0; i < m.rows; i++) {
		//this.data[this.rows] = m.data[i];
		this.data.push(m.data[i]);
		this.rows++;
	}
  }
*/

  static split(m,rows) {
    if (m.rows % rows != 0) throw(new Error("Split Error!"));
	let result = new Matrix(m.rows/rows,1);
	let cnt = 0;
	while (cnt < result.rows) {
		result.data[cnt][0] = new Matrix(rows,m.cols);
		for (let i = 0; i < rows; i++) {
			result.data[cnt][0].data[i] = m.data[cnt*rows+i];
		}
		cnt++;
	}
	return result;
  }


  static toMatrix(src, dim) { 
	switch (true) {
		case (src instanceof p5.Vector):
			if (dim == 2) return new Matrix([
			  [src.x],
			  [src.y]
			]);
			
			if (dim == 3) return new Matrix([
			  [src.x],
			  [src.y],
			  [src.z]
			]);
			break;
		
		case (src instanceof p5.Image):
			return new Matrix(src);
			
	}
	
	return -1;
	
  }

  static to_p5Vect(m1) {
    if (m1.rows == 2) return new p5.Vector(m1.data[0][0], m1.data[1][0]);
    if (m1.rows == 3) return new p5.Vector(m1.data[0][0], m1.data[1][0], m1.data[2][0]);
  }

  static to_p5Image(m) {
	let r = m.rows;
	let c = m.cols;
	let img = createImage(c,r);
	img.loadPixels();
	for (let i = 0; i < m.rows; i++) {
		for (let j = 0; j < m.cols; j++) {
			let idx = 4 * (i * m.cols + j);
			img.pixels[idx + 0] = floor(m.data[i][j]);
			img.pixels[idx + 1] = floor(m.data[i][j]);
			img.pixels[idx + 2] = floor(m.data[i][j]);
			img.pixels[idx + 3] = 255;
		}
	}
	img.updatePixels();
	return img;
  }
  
  to_p5Image() {
	let r = m.rows;
	let c = m.cols;
	let img = createImage(r,c);
	img.loadPixels();
	for (let i = 0; i < this.rows; i++) {
		for (let j = 0; j < this.cols; j++) {
			let idx = 4 * (i * this.cols + j);
			img.pixels[idx + 0] = floor(this.data[i][j]);
			img.pixels[idx + 1] = floor(this.data[i][j]);
			img.pixels[idx + 2] = floor(this.data[i][j]);
			img.pixels[idx + 3] = 255;
		}
	}
	img.updatePixels();
	return img;
  }
  
  toVect() {
    if (this.rows == 2) return new p5.Vector(this.data[0][0], this.data[1][0]);
    if (this.rows == 3) return new p5.Vector(this.data[0][0], this.data[1][0], this.data[2][0]);
  }

  toArray(ser) {
    let result = [];
    if (this.rows == 1) {
      for (let j = 0; j < this.cols; j++) {
        result.push(this.data[0][j]);
      }
    } else if (this.cols == 1) {
      for (let i = 0; i < this.rows; i++) {
        result.push(this.data[i][0]);
      }
    } else if (ser) {
      for (let i = 0; i < this.rows; i++) {
        for (let j = 0; j < this.cols; j++) {
          result.push(this.data[i][j]);
        }
      }
    } else {
      for (let i = 0; i < this.rows; i++) {
        result[i] = [];
        for (let j = 0; j < this.cols; j++) {
          result[i].push(this.data[i][j]);
        }
      }
    }
    return result;
  }

  static softmax(m) {
    let sum = 0;
    let result = new Matrix(m.rows, m.cols);
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        sum += exp(m.data[i][j]);
      }
    }
    for (let i = 0; i < m.rows; i++) {
      for (let j = 0; j < m.cols; j++) {
        result.data[i][j] = exp(m.data[i][j]) / sum;
      }
    }
    return result;
  }

  softmax() {
  	let sum = 0;
  	for (let i = 0; i < this.rows; i++) {
  	  for (let j = 0; j < this.cols; j++) {
  		sum += exp(this.data[i][j]);
  	  }
  	}
  	for (let i = 0; i < this.rows; i++) {
  	  for (let j = 0; j < this.cols; j++) {
  		this.data[i][j] = exp(this.data[i][j]) / sum;
  	  }
  	}
  }

  inv() {

    if (this.rows != this.cols) {
      throw (new Error("Inverse Matrix can only be calculated for square matrices!"))
    }

    let n = this.rows;
    let result = new Matrix(n, n, "unit");

    for (let p = 0; p < n-1 ; p++) {
      for (let i = p + 1; i < n; i++) {
        let factor = -1 * (this.data[i][p] / this.data[p][p]);
        //console.log("factor: " + factor + "; p: " + p);

        for (let j = 0; j < n; j++) {
          this.data[i][j] += my_round(factor * this.data[p][j]);
          result.data[i][j] += my_round(factor * result.data[p][j]);
        }
        // this.print();
        // result.print();
      }
    }
    //console.log("lower left corner all zero");
    for (let p = n - 1; p > 0; p--) {
      for (let i = p - 1; i >= 0; i--) {
        let factor = -1 * (this.data[i][p] / this.data[p][p]);
        //console.log("factor: " + factor + "; p: " + p + "; i: " + i);
        for (let j = 0; j < n; j++) {
          this.data[i][j] += my_round(factor * this.data[p][j]);
          result.data[i][j] += my_round(factor * result.data[p][j]);
        }
        // this.print();
        // result.print();
      }
    }
    //console.log("upper right corner all zero");

    for (let i = 0; i < n; i++) {
      let factor = 1 / this.data[i][i];
      //console.log("factor: " + factor);

      for (let j = 0; j < n; j++) {
        this.data[i][j] = my_round(this.data[i][j] * factor,12);
        result.data[i][j] = my_round(result.data[i][j] * factor,12);;
      }
      // this.print();
      // result.print();
    }

    this.data = result.data;

  }

/*
  //  this one didn't work
  // inv() {
  //
  //   if (this.rows != this.cols) {
  //     throw ("Inverse Matrix can only be calculated for n*n matrices!")
  //   }
  //
  //   let n = this.rows;
  //   let result = new Matrix(n, n, "zero");
  //
  //   let p = 0;
  //   let d = 1;
  //   let pivot;
  //
  //   while (p < n) {
  //     this.print();
  //     pivot = this.data[p][p];
  //     console.log("p: " + p + "; pivot: " + pivot);
  //     if (pivot == 0) {
  //       return -1;
  //       break;
  //     }
  //     d *= pivot;
  //     console.log("d: " + d + "; pivot: " + pivot);
  //     for (let j = 0; j < n; j++) {
  //       if (j != p) {
  //         this.data[p][j] = this.data[p][j] / pivot;
  //       }
  //     }
  //     for (let i = 0; i < n; i++) {
  //       if (i != p) {
  //         this.data[i][p] = -1 * (this.data[i][p] / pivot);
  //       }
  //     }
  //
  //     for (let i = 0; i < n; i++) {
  //       for (let j = 0; j < n; j++) {
  //         console.log("i: " + i + "; j: " + j);
  //         if (i != p && j != p) {
  //           this.data[i][j] = this.data[i][j] + this.data[p][j] * this.data[i][p];
  //           this.print();
  //         }
  //       }
  //     }
  //
  //     // result.print();
  //     this.data[p][p] = 1 / this.data[p][p];
  //     // this.print();
  //
  //     p++;
  //     // this.data = result.data;
  //   }
  // }
  //
  */
  
  print() {
    console.table(this.data);
  }
}
