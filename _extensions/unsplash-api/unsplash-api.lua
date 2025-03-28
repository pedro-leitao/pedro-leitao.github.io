local mimeImgExts = {
  ["image/jpeg"]="jpg",
  ["image/gif"]="gif",
  ["image/vnd.microsoft.icon"]="ico",
  ["image/avif"]="avif",
  ["image/bmp"]="bmp",
  ["image/png"]="png",
  ["image/svg+xml"]="svg",
  ["image/tiff"]="tif",
  ["image/webp"]="webp",
}

local function file_exists(name)
  local f = io.open(name, 'r')
  if f then
    io.close(f)
    return true
  else
    return false
  end
end

local function write_file(path, contents, mode)
  pandoc.system.make_directory(pandoc.path.directory(path), true)
  mode = mode or "a"
  local file = io.open(path, mode)
  if file then
    file:write(contents)
    file:close()
    return true
  else
    return false
  end
end

return {
  ['unsplash'] = function(args, kwargs, meta)
    -- usage examples:
    --   {{< unsplash cat >}}
    --   {{< unsplash keywords="cats" orientation="landscape" width="500" height="150" >}}
    
    local height = nil
    local width = nil
    local keywords = nil
    local classes = nil
    local float = nil
    local filename = nil
    local orientation = nil

    if args[1] ~= nil then
      filename = "unsplash-" .. pandoc.utils.stringify(args[1])
      local stem = pandoc.path.split_extension(pandoc.path.filename(filename))
      keywords = stem
    end

    if kwargs['height'] ~= nil and #kwargs['height'] > 0 then
      height = pandoc.utils.stringify(kwargs['height'])
    end

    if kwargs['width'] ~= nil and #kwargs['width'] > 0 then
      width = pandoc.utils.stringify(kwargs['width'])
    end

    if kwargs['keywords'] ~= nil and #kwargs['keywords'] > 0 then
      keywords = pandoc.utils.stringify(kwargs['keywords'])
    end

    if kwargs['orientation'] ~= nil and #kwargs['orientation'] > 0 then
      orientation = pandoc.utils.stringify(kwargs['orientation'])
    end

    if kwargs['class'] ~= nil and #kwargs['class'] > 0 then
      classes = pandoc.utils.stringify(kwargs['class'])
    end

    if kwargs['float'] ~= nil and #kwargs['float'] > 0 then
      float = pandoc.utils.stringify(kwargs['float'])
    end

    local imgContainer = function(imgEl)
      if quarto.doc.is_format("html") then
        quarto.doc.add_html_dependency({
          name = "unsplash-styles",
          version = "1.0.0",
          stylesheets = {"style.css"}
        })

        local style = ""
        if height then
          style = style .. "height: " .. height .. "; "
        end
        if width then
          style = style .. "width: " .. width .. "; "
        end

        local divAttrRaw = {}
        if style ~= "" then
          divAttrRaw['style'] = style
        end

        local clz = pandoc.List({"unsplash-container"})
        if float then
          clz:insert("float-" .. float)
        end

        if classes then
          for token in string.gmatch(classes, "[^%s]+") do
            clz:insert(token)
          end
        end

        local divAttr = pandoc.Attr("", clz, divAttrRaw)
        local div = pandoc.Div(imgEl, divAttr)
        return div
      else
        if height then
          imgEl.attr.attributes['height'] = height
        end
        if width then
          imgEl.attr.attributes['width'] = width
        end

        if classes then
          for token in string.gmatch(classes, "[^%s]+") do
            imgEl.attr.classes:insert(token)
          end
        end
        return imgEl
      end
    end

    -- Read Unsplash API key from metadata
    local api_key = pandoc.utils.stringify(meta["unsplash.client_id"])
    if not api_key then
      error("Missing 'client_id' key in 'unsplash' metadata.")
    end

    -- Build the Unsplash API URL for a random photo
    local api_url = "https://api.unsplash.com/photos/random?client_id=" .. api_key
    if keywords then
      api_url = api_url .. "&query=" .. keywords
    end
    if orientation then
      api_url = api_url .. "&orientation=" .. orientation
    end

    -- Fetch image metadata (JSON) from the Unsplash API
    local respMime, respContent = pandoc.mediabag.fetch(api_url)
    if not respContent then
      error("Failed to fetch image metadata from Unsplash API.")
    end
    if respMime ~= "application/json" then
      error("Unexpected response MIME type from Unsplash API.")
    end
    local respJson = pandoc.json.decode(respContent)
    local image_url = respJson.urls.raw

    -- Append width and height parameters if provided
    if width then
      image_url = image_url .. (string.find(image_url, "?") and "&" or "?") .. "w=" .. width
    end
    if height then
      image_url = image_url .. (string.find(image_url, "?") and "&" or "?") .. "h=" .. height
    end
    -- If both dimensions are provided, add cropping parameter
    if width and height then
      image_url = image_url .. (string.find(image_url, "?") and "&" or "?") .. "fit=crop"
    end

    -- Fetch the actual image using the constructed URL
    local imgMime, imgContents = pandoc.mediabag.fetch(image_url)
    if not imgContents then
      error("Failed to fetch image content from Unsplash.")
    end

    if filename and file_exists(filename) then
      return imgContainer(pandoc.Image("", filename))
    elseif filename then
      write_file(filename, imgContents, "wb")
      return imgContainer(pandoc.Image("", filename))
    else
      local tmpFileName = pandoc.path.filename(os.tmpname()) .. "." .. mimeImgExts[imgMime]
      pandoc.mediabag.insert(tmpFileName, imgMime, imgContents)
      return imgContainer(pandoc.Image("", tmpFileName))
    end
  end
}