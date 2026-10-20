// the api reference page: swagger-ui, served from this origin, pointed at the app's own schema.
// kept out of the html so the page needs no inline script (content security policy: script-src 'self')
window.addEventListener("DOMContentLoaded", () => {
  window.ui = SwaggerUIBundle({
    url: "/api/openapi.json",
    dom_id: "#swagger-ui",
    deepLinking: true,
    defaultModelsExpandDepth: 0,
    displayRequestDuration: true,
    tryItOutEnabled: false,
    presets: [SwaggerUIBundle.presets.apis],
    layout: "BaseLayout",
  });
});
