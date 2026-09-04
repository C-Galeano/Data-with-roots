// Highlights the navbar link that matches the current page.
document.addEventListener("DOMContentLoaded", () => {
    const currentPath = window.location.pathname;

    document.querySelectorAll(".navbar-nav .nav-link, .dropdown-item").forEach((link) => {
        const linkPath = new URL(link.href, window.location.origin).pathname;
        if (linkPath === currentPath) {
            link.classList.add("active");
        }
    });
});
