function playVideo(id) {
  const video = document.getElementById(id);
  video.play();
}

function changeMedia(id, ext) {
  const selectParent = document.getElementById('select-' + id);
  if (selectParent === null) {
      return ;
  }
  let path0 = selectParent.getElementsByTagName('select')[0].selectedOptions[0];
  let path1 = selectParent.getElementsByTagName('select')[1].selectedOptions[0];
  path0 = path0.textContent.trim().toLowerCase();
  path0 = path0.split(/[\s-]+/)[0] || '';
  path1 = path1.textContent.trim().toLowerCase();

  const media = document.getElementById(id);
  let mediaSrc;
  if (media.tagName === 'VIDEO') {
    mediaSrc = media.querySelector('source').src;
  } else {
    mediaSrc = media.src;
  }
  
  const mediaPath = mediaSrc.split("/");
  const videoBase = mediaPath[mediaPath.length-1].replace(ext, "").split("_");
  let newmediaPath = mediaPath.slice(0, -1).join("/") + "/";

  newmediaPath += path0 + "_" + path1 + "." + ext;
  media.src = newmediaPath;
  
  if (media.tagName === 'VIDEO') {
    media.load();
  }
}

function showTaskRow(id) {
  var selectParent = document.getElementById('select-' + id);
  if (selectParent === null) {
    return ;
  }
  let task = selectParent.getElementsByTagName('select')[0].selectedOptions[0];
  task = task.textContent.trim().toLowerCase();
  var rows = document.getElementsByTagName('tr');
  for (var i = 0; i < rows.length; i++) {
    if (rows[i].id !== "header") {
      rows[i].style.display = "none";
    }
  }
  document.getElementById(task).style.display = "table-row";
}