#![cfg(target_arch = "wasm32")]
#![allow(dead_code, unused_extern_crates, unused_imports)]
use wasm_bindgen::prelude::*;
use winit::{
  event::{Event, WindowEvent},
  event_loop::{ControlFlow, EventLoop},
  window::Window,
};

use std::cell::{Cell, RefCell};

use gfx_backend_gl as back;
use gfx_hal::{
  self as hal, buffer, command, format as f,
  format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle},
  image as i, memory as m, pass,
  pass::Subpass,
  pool,
  prelude::*,
  pso,
  pso::{PipelineStage, ShaderStageFlags, VertexInputRate},
  queue::{QueueGroup, Submission},
  window,
};

use crate::renderer::Renderer;

#[wasm_bindgen(start)]
pub fn wasm_setup() {
  set_panic_hook();

  console_log::init_with_level(log::Level::Info)
    .expect(&format!("could not setup logging backend"));
}

fn set_panic_hook() {
  // When the `console_error_panic_hook` feature is enabled, we can call the
  // `set_panic_hook` function at least once during initialization, and then
  // we will get better error messages if our code ever panics.
  //
  // For more details see
  // https://github.com/rustwasm/console_error_panic_hook#readme
  console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct WasmPlatform {
  event_loop: Option<EventLoop<()>>,
  window: Window,
  renderer: Option<Renderer<back::Backend>>,
}

#[wasm_bindgen]
impl WasmPlatform {
  #[wasm_bindgen(constructor)]
  pub fn new(canvas: web_sys::HtmlCanvasElement) -> Result<WasmPlatform, JsValue> {
    use winit::platform::web::{WindowBuilderExtWebSys, WindowExtWebSys};
    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::WindowBuilder::new()
      .with_inner_size(winit::dpi::LogicalSize {
        width: 480,
        height: 480,
      })
      .with_canvas(Some(canvas.clone()))
      .build(&event_loop)
      .or_else(|e| Err(js_sys::Error::new(&format!("hi, {:?}", e))))?;

    let mut plt = WasmPlatform {
      window,
      event_loop: Some(event_loop),
      renderer: None,
    };
    plt
      .setup_gfx()
      .or_else(|e| Err(js_sys::Error::new(&format!("hi, {:?}", e))))?;
    Ok(plt)
  }

  fn setup_gfx(&mut self) -> Result<(), failure::Error> {
    assert!(self.renderer.is_none());
    let surface = back::Surface::from_raw_handle(&self.window);
    let mut adapters: Vec<_> = surface.enumerate_adapters();
    failure::ensure!(adapters.len() != 0);
    let adapter = adapters.remove(0);
    let dims = self.window.inner_size();
    let renderer = Renderer::new(
      None,
      surface,
      adapter,
      window::Extent2D {
        width: dims.width,
        height: dims.height,
      },
    )?;
    self.renderer = Some(renderer);
    Ok(())
  }

  pub fn run(mut self) {
    let mut event_loop_opt = None;
    std::mem::swap(&mut self.event_loop, &mut event_loop_opt);
    let renderer = match self.renderer {
      Some(r) => r,
      _ => panic!("renderer is missing"),
    };
    self.renderer = None;
    match event_loop_opt {
      Some(event_loop) => event_loop.run(move |event, _, control_flow: &mut ControlFlow| {
        *control_flow = ControlFlow::Wait;
        match event {
          Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            window_id,
          } if window_id == self.window.id() => *control_flow = ControlFlow::Exit,
          Event::MainEventsCleared => {
            self.window.request_redraw();
          }
          _ => (),
        }
      }),
      _ => panic!("Platform.event_loop is None"),
    }
  }
}
