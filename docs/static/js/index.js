// ESSENTIAL project page — minimal interactions.
// (Removed unused Nerfies carousel/slider/interpolation code that referenced
//  undefined globals and threw ReferenceError in the console.)

$(document).ready(function () {
  // Mobile navbar burger toggle (harmless no-op if the burger is absent).
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  // Highlight the active section in the sticky nav while scrolling.
  var $navLinks = $(".section-nav a");
  var sections = $navLinks
    .map(function () {
      var id = $(this).attr("href");
      return id && id.charAt(0) === "#" && document.querySelector(id) ? id : null;
    })
    .get();

  function onScroll() {
    var pos = window.scrollY + 120;
    var current = sections[0];
    for (var i = 0; i < sections.length; i++) {
      var el = document.querySelector(sections[i]);
      if (el && el.offsetTop <= pos) current = sections[i];
    }
    $navLinks.removeClass("is-active");
    $navLinks.filter('[href="' + current + '"]').addClass("is-active");
  }
  if (sections.length) {
    $(window).on("scroll", onScroll);
    onScroll();
  }
});

// Copy the BibTeX entry to the clipboard.
function copyBibtex() {
  var el = document.getElementById("bibtex-content");
  if (!el) return;
  var text = el.innerText;
  var btn = document.getElementById("bibtex-copy-btn");
  var done = function () {
    if (!btn) return;
    var prev = btn.innerText;
    btn.innerText = "Copied!";
    btn.classList.add("is-copied");
    setTimeout(function () {
      btn.innerText = prev;
      btn.classList.remove("is-copied");
    }, 1500);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(done);
  } else {
    // Fallback for non-secure contexts / older browsers.
    var ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta);
    done();
  }
}
