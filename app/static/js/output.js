var el = x => document.getElementById(x);

var img_enc = document.getElementById('img_b64');

// console.log(img_enc);

var img_blob = dataURLtoBlob(img_enc.value);

// console.log(img_blob);

var url = URL.createObjectURL(img_blob);
var blobAnchor = document.getElementById('myButton');
blobAnchor.download = 'enhanced_image.png';
blobAnchor.href = url;
blobAnchor.target = "_blank"

var image_box = document.getElementById('myImage')
var url2 = URL.createObjectURL(img_blob);
image_box.src = url2

// document.getElementById('myButton').href = img_blob;
// document.getElementById('myButton').download = 'test.png';

// document.getElementById('myButton').onclick = function(event){

//     url = window.URL.createObjectURL(img_blob);
//     this.href = url;
//     this.target = '_blank';
// }



// var url = URL.createObjectURL(blob);

// var blobAnchor = document.getElementById('blob');
// var dataURIAnchor = document.getElementById('dataURI');

// var dataURI_DL = function() {

//     var url = URL.createObjectURL(img_blob);
//     var blobAnchor = document.getElementById('blob');
//     var dataURIAnchor = document.getElementById('dataURI');
//     blobAnchor.download = dataURIAnchor.download = 'test.png';
//     blobAnchor.href = url;
//     dataURIAnchor.href = dataURI;
//     stat_.textContent = '';
  
//     blobAnchor.onclick = function() {
//       requestAnimationFrame(function() {
//         URL.revokeObjectURL(url);
//       })
//     };


// }


function dataURLtoBlob(dataURL) {
    // Decode the dataURL
    var imageType = dataURL.split(',')[0];     
    var binary = atob(dataURL.split(',')[1]);
    // Create 8-bit unsigned array
    var array = [];
    for(var i = 0; i < binary.length; i++) {
        array.push(binary.charCodeAt(i));
    }
    // Return our Blob object
    // console.log("imageType", imageType);

    if (imageType.indexOf("jpeg") >=0) {
        return new Blob([new Uint8Array(array)], {type: 'image/jpeg'});
    }
    else {
        return new Blob([new Uint8Array(array)], {type: 'image/png'});
    }
  }