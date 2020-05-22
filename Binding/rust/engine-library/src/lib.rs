extern crate libc;

mod application {
    use winit::{
        event::{Event, WindowEvent},
        event_loop::{ControlFlow, EventLoop},
        window::{Window, WindowBuilder},
    };

    pub struct RustApplication {
        event_loop: EventLoop<()>,
        window: Window,
        is_quit: bool,
    }

    impl RustApplication {
        #[no_mangle]
        pub extern "C" fn initialize(&mut self) -> libc::c_int {
            self.event_loop = EventLoop::new();
            self.window = WindowBuilder::new()
                    .with_title("Hello Engine!")
                    .build(&self.event_loop).unwrap();
            0
        }

        #[no_mangle]
        pub extern "C" fn finalize(&mut self) {

        }

        #[no_mangle]
        pub extern "C" fn tick(self) {
            self.event_loop.run(move |event, _, control_flow| {
                *control_flow = ControlFlow::Wait;
        
                match event {
                    Event::WindowEvent {
                        event: WindowEvent::CloseRequested,
                        window_id,
                    } if window_id == self.window.id() => *control_flow = ControlFlow::Exit,
                    _ => (),
                }
            });
        }

        #[no_mangle]
        pub extern "C" fn is_quit(&self) -> bool {
            self.is_quit
        }

        #[no_mangle]
        pub extern "C" fn get_main_window_handler() -> *mut libc::c_void {
            std::ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
