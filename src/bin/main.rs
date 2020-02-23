#![cfg_attr(
    not(any(
        feature = "vulkan",
        feature = "dx11",
        feature = "dx12",
        feature = "metal",
        feature = "gl",
        feature = "wgl"
    )),
    allow(dead_code, unused_extern_crates, unused_imports)
)]
use failure::format_err;

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

#[cfg(feature = "dx11")]
extern crate gfx_backend_dx11 as back;
#[cfg(feature = "dx12")]
extern crate gfx_backend_dx12 as back;
#[cfg(any(feature = "gl", feature = "wgl"))]
extern crate gfx_backend_gl as back;
#[cfg(feature = "metal")]
extern crate gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
extern crate gfx_backend_vulkan as back;

use webgl_gfx::renderer::Renderer;

fn main() -> Result<(), failure::Error> {
    #[cfg(target_arch = "wasm32")]
    console_log::init_with_level(log::Level::Debug).unwrap();
    #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
    env_logger::init();

    let event_loop = winit::event_loop::EventLoop::new();
    let (width, height) = (640, 640);

    let wb = winit::window::WindowBuilder::new()
        .with_min_inner_size(winit::dpi::Size::Logical(winit::dpi::LogicalSize::new(
            64.0, 64.0,
        )))
        .with_inner_size(winit::dpi::Size::Physical(winit::dpi::PhysicalSize::new(
            width, height,
        )))
        .with_title("quad".to_string());

    // instantiate backend
    #[cfg(not(feature = "gl"))]
    let (_window, instance, mut adapters, surface) = {
        let window = wb.build(&event_loop).unwrap();
        let instance = back::Instance::create("gfx-rs quad", 1)
            .or(Err(format_err!("Failed to create an instance!")))?;
        let surface = unsafe {
            instance
                .create_surface(&window)
                .or(Err(format_err!("Failed to create a surface!")))?
        };
        let adapters = instance.enumerate_adapters();
        // Return `window` so it is not dropped: dropping it invalidates `surface`.
        (window, Some(instance), adapters, surface)
    };
    #[cfg(feature = "gl")]
    let (_window, instance, mut adapters, surface) = {
        #[cfg(not(target_arch = "wasm32"))]
        let (window, surface) = {
            let builder =
                back::config_context(back::glutin::ContextBuilder::new(), ColorFormat::SELF, None)
                    .with_vsync(true);
            let windowed_context = builder.build_windowed(wb, &event_loop)?;
            let (context, window) = unsafe {
                windowed_context
                    .make_current()
                    .expect("Unable to make context current")
                    .split()
            };
            let surface = back::Surface::from_context(context);
            (window, surface)
        };
        #[cfg(target_arch = "wasm32")]
        let (window, surface) = {
            let window = wb.build(&event_loop)?;
            web_sys::window()
                .unwrap()
                .document()?
                .body()?
                .append_child(&winit::platform::web::WindowExtWebSys::canvas(&window));
            let surface = back::Surface::from_raw_handle(&window);
            (window, surface)
        };

        let adapters = surface.enumerate_adapters();
        (window, None, adapters, surface)
    };

    for adapter in &adapters {
        println!("{:?}", adapter.info);
    }

    let adapter = adapters.remove(0);

    let mut renderer = Renderer::new(instance, surface, adapter, window::Extent2D {width, height})?;

    renderer.render();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Wait;
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = winit::event_loop::ControlFlow::Exit
                }
                winit::event::WindowEvent::KeyboardInput {
                    input:
                        winit::event::KeyboardInput {
                            virtual_keycode: Some(winit::event::VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = winit::event_loop::ControlFlow::Exit,
                winit::event::WindowEvent::Resized(dims) => {
                    println!("resized to {:?}", dims);
                    #[cfg(all(feature = "gl", not(target_arch = "wasm32")))]
                    {
                        let context = renderer.surface.context();
                        context.resize(dims);
                    }
                    renderer.recreate_swapchain(window::Extent2D {
                        width: dims.width,
                        height: dims.height,
                    }).expect("could not resize window");
                }
                _ => {}
            },
            winit::event::Event::RedrawEventsCleared => {
                renderer.render();
            }
            _ => {}
        }
    });
}
