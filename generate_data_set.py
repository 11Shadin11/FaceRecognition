"""Launch the application to add samples from the camera or an imported video.

Sample collection is implemented in face_app.service.FaceService.collect_samples.
Importing this module does not create windows or start processing.
"""


if __name__ == "__main__":
    from face_app.main import main
    raise SystemExit(main())
