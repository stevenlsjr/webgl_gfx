use failure::Error;
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
use std::{
  io::Cursor,
  iter,
  mem::{self, ManuallyDrop},
  ptr,
};

pub struct Vertex {
  position: [f32; 2],
  uv: [f32; 2],
}

const QUAD_VERTS: [Vertex; 6] = [
  Vertex {
    position: [-0.5, 0.33],
    uv: [0.0, 1.0],
  },
  Vertex {
    position: [0.5, 0.33],
    uv: [1.0, 1.0],
  },
  Vertex {
    position: [0.5, -0.33],
    uv: [1.0, 0.0],
  },
  Vertex {
    position: [-0.5, 0.33],
    uv: [0.0, 1.0],
  },
  Vertex {
    position: [0.5, -0.33],
    uv: [1.0, 0.0],
  },
  Vertex {
    position: [-0.5, -0.33],
    uv: [0.0, 0.0],
  },
];

const COLOR_RANGE: i::SubresourceRange = i::SubresourceRange {
  aspects: f::Aspects::COLOR,
  levels: 0..1,
  layers: 0..1,
};

pub struct Renderer<B: hal::Backend> {
  instance: Option<B::Instance>,
  device: B::Device,
  queue_group: QueueGroup<B>,
  desc_pool: ManuallyDrop<B::DescriptorPool>,
  surface: ManuallyDrop<B::Surface>,
  adapter: hal::adapter::Adapter<B>,
  format: hal::format::Format,
  dimensions: window::Extent2D,
  viewport: pso::Viewport,
  render_pass: ManuallyDrop<B::RenderPass>,
  pipeline: ManuallyDrop<B::GraphicsPipeline>,
  pipeline_layout: ManuallyDrop<B::PipelineLayout>,
  desc_set: B::DescriptorSet,
  set_layout: ManuallyDrop<B::DescriptorSetLayout>,
  submission_complete_semaphores: Vec<B::Semaphore>,
  submission_complete_fences: Vec<B::Fence>,
  cmd_pools: Vec<B::CommandPool>,
  cmd_buffers: Vec<B::CommandBuffer>,
  vertex_buffer: ManuallyDrop<B::Buffer>,
  image_upload_buffer: ManuallyDrop<B::Buffer>,
  image_logo: ManuallyDrop<B::Image>,
  image_srv: ManuallyDrop<B::ImageView>,
  buffer_memory: ManuallyDrop<B::Memory>,
  image_memory: ManuallyDrop<B::Memory>,
  image_upload_memory: ManuallyDrop<B::Memory>,
  sampler: ManuallyDrop<B::Sampler>,
  frames_in_flight: usize,
  frame: u64,
}

impl<B: hal::Backend> Renderer<B> {
  pub fn new(
    instance: Option<B::Instance>,
    mut surface: B::Surface,
    adapter: hal::adapter::Adapter<B>,
    initial_dimensions: window::Extent2D,
  ) -> Result<Renderer<B>, Error> {
    let memory_types = adapter.physical_device.memory_properties().memory_types;
    let limits = adapter.physical_device.limits();
    let family = adapter
      .queue_families
      .iter()
      .find(|family| {
        surface.supports_queue_family(family) && family.queue_type().supports_graphics()
      })
      .ok_or(failure::format_err!("could not find valid queue family"))?;

    let mut gpu = unsafe {
      adapter
        .physical_device
        .open(&[(family, &[1.0])], gfx_hal::Features::empty())
    }?;
    let mut queue_group = gpu
      .queue_groups
      .pop()
      .ok_or(failure::format_err!("could not find gpu queue group"))?;
    let device = gpu.device;
    let mut command_pool = unsafe {
      device.create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())
    }?;
    let set_layout = ManuallyDrop::new(unsafe {
      device.create_descriptor_set_layout(
        &[
          pso::DescriptorSetLayoutBinding {
            binding: 0,
            ty: pso::DescriptorType::SampledImage,
            count: 1,
            stage_flags: ShaderStageFlags::FRAGMENT,
            immutable_samplers: false,
          },
          pso::DescriptorSetLayoutBinding {
            binding: 1,
            ty: pso::DescriptorType::Sampler,
            count: 1,
            stage_flags: ShaderStageFlags::FRAGMENT,
            immutable_samplers: false,
          },
        ],
        &[],
      )
    }?);

    let mut desc_pool = ManuallyDrop::new(unsafe {
      device.create_descriptor_pool(
        1, // sets
        &[
          pso::DescriptorRangeDesc {
            ty: pso::DescriptorType::SampledImage,
            count: 1,
          },
          pso::DescriptorRangeDesc {
            ty: pso::DescriptorType::Sampler,
            count: 1,
          },
        ],
        pso::DescriptorPoolCreateFlags::empty(),
      )
    }?);
    let desc_set = unsafe { desc_pool.allocate_set(&set_layout) }?;
    let non_coherent_alignment = limits.non_coherent_atom_size as u64;
    let buffer_stride = mem::size_of::<Vertex>() as u64;
    let buffer_len = QUAD_VERTS.len() as u64 * buffer_stride;
    assert_ne!(buffer_len, 0);

    let padded_buffer_len =
      ((buffer_len + non_coherent_alignment - 1) / non_coherent_alignment) * non_coherent_alignment;
    let mut vertex_buffer =
      ManuallyDrop::new(unsafe { device.create_buffer(padded_buffer_len, buffer::Usage::VERTEX) }?);
    let buffer_req = unsafe { device.get_buffer_requirements(&vertex_buffer) };
    let upload_type = memory_types
      .iter()
      .enumerate()
      .position(|(id, mem_type)| {
        buffer_req.type_mask & (1 << id) != 0
          && mem_type.properties.contains(m::Properties::CPU_VISIBLE)
      })
      .ok_or(failure::format_err!("could not find buffer upload type"))?;
    let upload_type = upload_type.into();
    let buffer_memory = unsafe {
      let memory = device.allocate_memory(upload_type, buffer_req.size)?;
      device.bind_buffer_memory(&memory, 0, &mut vertex_buffer)?;
      let mapping = device.map_memory(&memory, 0..padded_buffer_len)?;
      ptr::copy_nonoverlapping(
        QUAD_VERTS.as_ptr() as *const u8,
        mapping,
        buffer_len as usize,
      );
      device.flush_mapped_memory_ranges(iter::once((&memory, 0..padded_buffer_len)))?;
      device.unmap_memory(&memory);
      ManuallyDrop::new(memory)
    };

    let img_data = include_bytes!("data/image.png");

    let img =
      image::load(Cursor::new(&img_data[..]), image::ImageFormat::Png).map(|i| i.to_rgba())?;
    let (width, height) = img.dimensions();
    let kind = i::Kind::D2(width as i::Size, height as i::Size, 1, 1);
    let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
    let image_stride = 4usize;
    let row_pitch = (width * image_stride as u32 + row_alignment_mask) & !row_alignment_mask;
    let upload_size = (height * row_pitch) as u64;
    let padded_upload_size = ((upload_size + non_coherent_alignment - 1) / non_coherent_alignment)
      * non_coherent_alignment;

    let mut image_upload_buffer = ManuallyDrop::new(unsafe {
      device.create_buffer(padded_upload_size, buffer::Usage::TRANSFER_SRC)
    }?);
    let image_mem_reqs = unsafe { device.get_buffer_requirements(&image_upload_buffer) };

    // copy image data into staging buffer
    let image_upload_memory = unsafe {
      let memory = device.allocate_memory(upload_type, image_mem_reqs.size)?;
      device.bind_buffer_memory(&memory, 0, &mut image_upload_buffer)?;
      let mapping = device.map_memory(&memory, 0..padded_upload_size).unwrap();
      for y in 0..height as usize {
        let row =
          &(*img)[y * (width as usize) * image_stride..(y + 1) * (width as usize) * image_stride];
        ptr::copy_nonoverlapping(
          row.as_ptr(),
          mapping.offset(y as isize * row_pitch as isize),
          width as usize * image_stride,
        );
      }
      device
        .flush_mapped_memory_ranges(iter::once((&memory, 0..padded_upload_size)))
        .unwrap();
      device.unmap_memory(&memory);
      ManuallyDrop::new(memory)
    };

    let mut image_logo = ManuallyDrop::new(
      unsafe {
        device.create_image(
          kind,
          1,
          ColorFormat::SELF,
          i::Tiling::Optimal,
          i::Usage::TRANSFER_DST | i::Usage::SAMPLED,
          i::ViewCapabilities::empty(),
        )
      }
      .unwrap(),
    );
    let image_req = unsafe { device.get_image_requirements(&image_logo) };

    let device_type = memory_types
      .iter()
      .enumerate()
      .position(|(id, memory_type)| {
        image_req.type_mask & (1 << id) != 0
          && memory_type.properties.contains(m::Properties::DEVICE_LOCAL)
      })
      .unwrap()
      .into();
    let image_memory =
      ManuallyDrop::new(unsafe { device.allocate_memory(device_type, image_req.size) }.unwrap());

    unsafe { device.bind_image_memory(&image_memory, 0, &mut image_logo) }?;
    let image_srv = ManuallyDrop::new(unsafe {
      device.create_image_view(
        &image_logo,
        i::ViewKind::D2,
        ColorFormat::SELF,
        Swizzle::NO,
        COLOR_RANGE.clone(),
      )
    }?);

    let sampler = ManuallyDrop::new(
      unsafe { device.create_sampler(&i::SamplerDesc::new(i::Filter::Linear, i::WrapMode::Clamp)) }
        .or(Err(failure::format_err!("could not create image sampler")))?,
    );

    unsafe {
      device.write_descriptor_sets(vec![
        pso::DescriptorSetWrite {
          set: &desc_set,
          binding: 0,
          array_offset: 0,
          descriptors: Some(pso::Descriptor::Image(
            &*image_srv,
            i::Layout::ShaderReadOnlyOptimal,
          )),
        },
        pso::DescriptorSetWrite {
          set: &desc_set,
          binding: 1,
          array_offset: 0,
          descriptors: Some(pso::Descriptor::Sampler(&*sampler)),
        },
      ]);
    }

    // copy buffer to texture
    let mut copy_fence = device.create_fence(false)?;
    unsafe {
      let mut cmd_buffer = command_pool.allocate_one(command::Level::Primary);
      cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
      let image_barrier = m::Barrier::Image {
        states: (i::Access::empty(), i::Layout::Undefined)
          ..(i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal),
        target: &*image_logo,
        families: None,
        range: COLOR_RANGE.clone(),
      };
      cmd_buffer.pipeline_barrier(
        PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
        m::Dependencies::empty(),
        &[image_barrier],
      );
      cmd_buffer.copy_buffer_to_image(
        &image_upload_buffer,
        &image_logo,
        i::Layout::TransferDstOptimal,
        &[command::BufferImageCopy {
          buffer_offset: 0,
          buffer_width: row_pitch / (image_stride as u32),
          buffer_height: height as u32,
          image_layers: i::SubresourceLayers {
            aspects: f::Aspects::COLOR,
            level: 0,
            layers: 0..1,
          },
          image_offset: i::Offset { x: 0, y: 0, z: 0 },
          image_extent: i::Extent {
            width,
            height,
            depth: 1,
          },
        }],
      );
      let image_barrier = m::Barrier::Image {
        states: (i::Access::TRANSFER_WRITE, i::Layout::TransferDstOptimal)
          ..(i::Access::SHADER_READ, i::Layout::ShaderReadOnlyOptimal),
        target: &*image_logo,
        families: None,
        range: COLOR_RANGE.clone(),
      };
      cmd_buffer.pipeline_barrier(
        PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
        m::Dependencies::empty(),
        &[image_barrier],
      );
      cmd_buffer.finish();
      queue_group.queues[0].submit_without_semaphores(Some(&cmd_buffer), Some(&mut copy_fence));
      device
        .wait_for_fence(&copy_fence, !0)
        .map_err(|e| failure::format_err!("could not wait on fance {:?}", e))?;
    };
    unsafe {
      device.destroy_fence(copy_fence);
    }
    let caps = surface.capabilities(&adapter.physical_device);
    let formats = surface.supported_formats(&adapter.physical_device);
    log::info!("formats: {:?}", formats);
    let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
      formats
        .iter()
        .find(|format| format.base_format().1 == ChannelType::Srgb)
        .map(|format| *format)
        .unwrap_or(formats[0])
    });
    let swap_config = window::SwapchainConfig::from_caps(&caps, format, initial_dimensions);
    let extent = swap_config.extent;
    unsafe {
      surface.configure_swapchain(&device, swap_config)?;
    }
    let render_pass = {
      let attachment = pass::Attachment {
        format: Some(format),
        samples: 1,
        ops: pass::AttachmentOps::new(
          pass::AttachmentLoadOp::Clear,
          pass::AttachmentStoreOp::Store,
        ),
        stencil_ops: pass::AttachmentOps::DONT_CARE,
        layouts: i::Layout::Undefined..i::Layout::Present,
      };
      let subpass = pass::SubpassDesc {
        colors: &[(0, i::Layout::ColorAttachmentOptimal)],
        depth_stencil: None,
        inputs: &[],
        resolves: &[],
        preserves: &[],
      };
      ManuallyDrop::new(
        unsafe { device.create_render_pass(&[attachment], &[subpass], &[]) }
          .map_err(|e| failure::format_err!("could not create render pass: {:?}", e))?,
      )
    };
    let frames_in_flight = 3;
    let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
    let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);
    let mut cmd_pools = Vec::with_capacity(frames_in_flight);
    let mut cmd_buffers = Vec::with_capacity(frames_in_flight);
    cmd_pools.push(command_pool);
    for _ in 1..frames_in_flight {
      unsafe {
        cmd_pools.push(
          device
            .create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())
            .expect("Can't create command pool"),
        );
      }
    }

    for i in 0..frames_in_flight {
      submission_complete_semaphores.push(
        device
          .create_semaphore()
          .expect("Could not create semaphore"),
      );
      submission_complete_fences.push(device.create_fence(true).expect("Could not create fence"));
      cmd_buffers.push(unsafe { cmd_pools[i].allocate_one(command::Level::Primary) });
    }

    let pipeline_layout = ManuallyDrop::new(
      unsafe {
        device.create_pipeline_layout(
          iter::once(&*set_layout),
          &[(pso::ShaderStageFlags::VERTEX, 0..8)],
        )
      }
      .expect("Can't create pipeline layout"),
    );
    let pipeline = {
      let vs_module = {
        let spirv =
          pso::read_spirv(Cursor::new(&include_bytes!("data/quad.vert.spv")[..])).unwrap();
        unsafe { device.create_shader_module(&spirv) }.unwrap()
      };
      let fs_module = {
        let spirv =
          pso::read_spirv(Cursor::new(&include_bytes!("./data/quad.frag.spv")[..])).unwrap();
        unsafe { device.create_shader_module(&spirv) }.unwrap()
      };
      const ENTRY_NAME: &str = "main";
      let pipeline = {
        let (vs_entry, fs_entry) = (
          pso::EntryPoint {
            entry: ENTRY_NAME,
            module: &vs_module,
            specialization: hal::spec_const_list![0.8f32],
          },
          pso::EntryPoint {
            entry: ENTRY_NAME,
            module: &fs_module,
            specialization: pso::Specialization::default(),
          },
        );

        let shader_entries = pso::GraphicsShaderSet {
          vertex: vs_entry,
          hull: None,
          domain: None,
          geometry: None,
          fragment: Some(fs_entry),
        };

        let subpass = Subpass {
          index: 0,
          main_pass: &*render_pass,
        };

        let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
          shader_entries,
          pso::Primitive::TriangleList,
          pso::Rasterizer::FILL,
          &*pipeline_layout,
          subpass,
        );
        pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
          mask: pso::ColorMask::ALL,
          blend: Some(pso::BlendState::ALPHA),
        });
        pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
          binding: 0,
          stride: mem::size_of::<Vertex>() as u32,
          rate: VertexInputRate::Vertex,
        });

        pipeline_desc.attributes.push(pso::AttributeDesc {
          location: 0,
          binding: 0,
          element: pso::Element {
            format: f::Format::Rg32Sfloat,
            offset: 0,
          },
        });
        pipeline_desc.attributes.push(pso::AttributeDesc {
          location: 1,
          binding: 0,
          element: pso::Element {
            format: f::Format::Rg32Sfloat,
            offset: 8,
          },
        });

        unsafe { device.create_graphics_pipeline(&pipeline_desc, None) }
      };

      unsafe {
        device.destroy_shader_module(vs_module);
      }
      unsafe {
        device.destroy_shader_module(fs_module);
      }

      ManuallyDrop::new(pipeline.unwrap())
    };

    // Rendering setup
    let viewport = pso::Viewport {
      rect: pso::Rect {
        x: 0,
        y: 0,
        w: extent.width as _,
        h: extent.height as _,
      },
      depth: 0.0..1.0,
    };

    Ok(Renderer {
      instance,
      device,
      queue_group,
      desc_pool,
      surface: ManuallyDrop::new(surface),
      adapter,
      format,
      dimensions: initial_dimensions,
      viewport,
      render_pass,
      pipeline,
      pipeline_layout,
      desc_set,
      set_layout,
      submission_complete_semaphores,
      submission_complete_fences,
      cmd_pools,
      cmd_buffers,
      vertex_buffer,
      image_upload_buffer,
      image_logo,
      image_srv,
      buffer_memory,
      image_memory,
      image_upload_memory,
      sampler,
      frames_in_flight,
      frame: 0,
    })
  }

  pub fn render(&mut self) {}
  pub fn recreate_swapchain(&mut self, dims: window::Extent2D) -> Result<(), failure::Error> {
    Ok(())
  }
}

impl<B> Drop for Renderer<B>
where
  B: hal::Backend,
{
  fn drop(&mut self) {
    self.device.wait_idle().expect("Could not wait for device");
    unsafe {
      self
        .device
        .destroy_descriptor_pool(ManuallyDrop::into_inner(ptr::read(&self.desc_pool)));
      self
        .device
        .destroy_descriptor_set_layout(ManuallyDrop::into_inner(ptr::read(&self.set_layout)));
      self
        .device
        .destroy_buffer(ManuallyDrop::into_inner(ptr::read(&self.vertex_buffer)));
      self
        .device
        .destroy_buffer(ManuallyDrop::into_inner(ptr::read(
          &self.image_upload_buffer,
        )));
      self
        .device
        .destroy_image(ManuallyDrop::into_inner(ptr::read(&self.image_logo)));
      self
        .device
        .destroy_image_view(ManuallyDrop::into_inner(ptr::read(&self.image_srv)));
      self
        .device
        .destroy_sampler(ManuallyDrop::into_inner(ptr::read(&self.sampler)));
      for p in self.cmd_pools.drain(..) {
        self.device.destroy_command_pool(p);
      }
      for s in self.submission_complete_semaphores.drain(..) {
        self.device.destroy_semaphore(s);
      }
      for f in self.submission_complete_fences.drain(..) {
        self.device.destroy_fence(f);
      }
      self
        .device
        .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&self.render_pass)));
      self.surface.unconfigure_swapchain(&self.device);
      self
        .device
        .free_memory(ManuallyDrop::into_inner(ptr::read(&self.buffer_memory)));
      self
        .device
        .free_memory(ManuallyDrop::into_inner(ptr::read(&self.image_memory)));
      self.device.free_memory(ManuallyDrop::into_inner(ptr::read(
        &self.image_upload_memory,
      )));
      self
        .device
        .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.pipeline)));
      self
        .device
        .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(&self.pipeline_layout)));
      if let Some(instance) = &self.instance {
        let surface = ManuallyDrop::into_inner(ptr::read(&self.surface));
        instance.destroy_surface(surface);
      }
    }
  }
}
