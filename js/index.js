const canvas = document.querySelector('canvas#app-canvas');
import('../pkg/index.js')
  .then(({ WasmPlatform }) => {
    const app = new WasmPlatform(canvas);
    app.run();
  })
  .catch((e)=>{
    console.error(e)
  });
