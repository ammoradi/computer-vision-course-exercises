// Image upload handlings
let imgElement = document.getElementById('imageSrc');
let inputElement = document.getElementById('fileInput');
inputElement.addEventListener('change', (e) => {
    imgElement.src = URL.createObjectURL(e.target.files[0]);
}, false);
// After image upload
imgElement.onload = function () {
    let src = cv.imread(imgElement);
    let rgbaPlanes = new cv.MatVector();
    // Split the Mat
    cv.split(src, rgbaPlanes);
    // Get channels
    let R = rgbaPlanes.get(0);
    let G = rgbaPlanes.get(1);
    let B = rgbaPlanes.get(2);
    // Get channels lightings
    console.log(cv.mean(R)[0].toFixed(2))
    console.log(cv.mean(G)[0].toFixed(2))
    console.log(cv.mean(B)[0].toFixed(2))
    // Show Channels
    cv.imshow('redChannel',R);
    cv.imshow('greenChannel',G);
    cv.imshow('blueChannel',B);
    // Make memory free
    src.delete();
    rgbaPlanes.delete();
    R.delete();
    G.delete();
    B.delete();
};

function onOpenCvReady() {
    document.getElementById('status').innerHTML = 'OpenCV.js is ready.';
}