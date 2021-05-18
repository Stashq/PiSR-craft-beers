<script lang="ts">
	
	import { actual_beer} from './store.js';
	let beer;
	actual_beer.subscribe
	(value => {beer = value})
		
	let beer_name="PLACE_HOLDER"
	let img = "./tmp.png"
	let beer_desc = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec gravida tortor eget felis molestie, non luctus quam aliquam. Nunc et arcu est. Nulla at lacus lorem. Mauris faucibus bibendum magna vel facilisis. Integer ut ligula quis sem efficitur luctus. Pellentesque a lacus nec magna mollis dapibus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Etiam in nibh ac magna ultricies mattis. Sed elementum arcu et lacus sollicitudin, a finibus mauris faucibus. Mauris diam arcu, sollicitudin et tortor at, convallis euismod nunc. Donec tempus mauris metus, in tempor orci pharetra eget. Nam euismod massa non egestas lacinia. Ut rutrum quam at mauris interdum, nec dictum ex finibus. Maecenas sit amet congue ipsum. Sed volutpat nisi ut vehicula congue. Donec justo ante, elementum quis ullamcorper sit amet, venenatis vitae tellus."

</script>
	<div style="min-width: 100px;">
		{#if typeof beer !== "undefined"}
		<h1 style="color:white">{beer[0]}</h1>
		${console.log(beer)}
		<img src={beer[3]} alt="focused beer" class="beer_img">
		<div class="content-scroller">
			<p style="color:white">brewery: {beer[1]}</p>
			<p style="color:white">rating: {beer[2]}</p>
			<p style="color:white">Chmiele: {beer[9]}</p>
			<p style="color:white">DROŻDŻE: {beer[11]}</p>
			<p style="color:white">KONTRAKTOR: {beer[12]}</p>
			<p style="color:white">STYL: {beer[8]}</p>
			<p style="color:white">SŁODY: {beer[10]}</p>
			<p style="color:white">blg: {beer[5]}</p>
			<p style="color:white">alc: {beer[6]}</p>
			<p style="color:white">ibu: {beer[7]}</p>
			
			TUTAJ POWINNIEN BYĆ WYKRES
			
			<img src={img} >
		</div>
		{/if}	
	  </div>


<style>

	.content-scroller{
		overflow: scroll;
		height: 38%;
		overflow-x: hidden; /* Hide horizontal scrollbar */
		
	}
	.beer_img{
		width: 100%;
		height: 50%;
	}
</style>